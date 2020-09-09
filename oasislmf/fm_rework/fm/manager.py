from .queue import TerminableQueue
from .financial_structure import load_financial_structure, INPUT_STORAGE, TEMP_STORAGE, OUTPUT_STORAGE
from .stream import read_stream_header, queue_event_reader, read_event, queue_event_writer, EventWriter
from .compute import event_computer, compute_event
try:
    from .compute import ray_event_computer, numba_to_python
    from .queue import RayTerminableQueue
    import ray
except ImportError:
    pass


import sys
import concurrent.futures
import logging
logger = logging.getLogger(__name__)


def run(run_mode, **kwargs):
    if run_mode == 0:
        return run_synchronous(**kwargs)
    elif run_mode == 1:
        run_threaded(**kwargs)
    elif run_mode == 2:
        run_ray(**kwargs)
    else:
        raise ValueError(f"Unknow run_mode {run_mode}")


def run_threaded(allocation_rule, static_path, files_in, queue_in_size, files_out, queue_out_size, **kwargs):
    compute_queue, node_to_index, node_to_dependencies, node_to_profile, output_item_index, storage_to_len, options, profile = load_financial_structure(
        allocation_rule, static_path)
    sentinel = 'STOP'

    queue_in = TerminableQueue(maxsize=queue_in_size, sentinel=sentinel)
    queue_out = TerminableQueue(maxsize=queue_out_size, sentinel=sentinel)

    logger.info(f"starting, {files_in}, {files_out}")
    if files_in is None:
        inputs = [None]
    else:
        inputs = files_in

    if files_out is None:
        outputs = [None]
    else:
        outputs = files_out

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs) + len(outputs) + 1) as executor:
        reader_tasks = [executor.submit(queue_event_reader, queue_in, stream_in, node_to_index, storage_to_len[INPUT_STORAGE])
                        for stream_in in inputs]
        logger.info(f"reader_tasks started")
        computation_task = executor.submit(event_computer, queue_in, queue_out, compute_queue, node_to_index,
                                           node_to_dependencies, node_to_profile, storage_to_len,
                                           options, profile, sentinel)
        logger.info(f"computation_task started")
        writer_tasks = [
            executor.submit(queue_event_writer, queue_out, stream_out, output_item_index, sentinel) for
            stream_out in outputs]
        logger.info(f"writer_tasks started")

        reader_result = [t.result() for t in reader_tasks]

        logger.info(f"reader_tasks finished")

        queue_in.put(sentinel)
        computation_result = computation_task.result()
        logger.info(f"computation_task finished")
        for _ in writer_tasks:
            queue_out.put(sentinel)

        writer_result = [t.result() for t in writer_tasks]
        logger.info(f"writer_tasks finished")


def run_ray(allocation_rule, static_path, files_in, queue_in_size, files_out, queue_out_size, ray_address, **kwargs):
    """ray can only be use for compute as the stream interface is limited to the running machine"""
    ray.init(address=ray_address)
    computation_task = int(ray.available_resources()['CPU']) - 2
    compute_queue, node_to_index, node_to_dependencies, node_to_profile, output_item_index, storage_to_len, options, profile = load_financial_structure(
        allocation_rule, static_path)

    nb_obj = numba_to_python(compute_queue, node_to_index, node_to_dependencies, node_to_profile, storage_to_len,
                             options)
    py_compute_queue, py_node_to_index, py_node_to_dependencies, py_node_to_profile, py_storage_to_len, py_options = nb_obj

    sentinel = 'STOP'

    queue_in = RayTerminableQueue(maxsize=queue_in_size, sentinel=sentinel)
    queue_out = RayTerminableQueue(maxsize=queue_out_size, sentinel=sentinel)

    logger.info(f"starting, {files_in}, {files_out}")
    if files_in is None:
        inputs = [None]
    else:
        inputs = files_in

    if files_out is None:
        outputs = [None]
    else:
        outputs = files_out

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs) + len(outputs)) as executor:
        try:
            compute_task = [ray_event_computer.remote(queue_in, queue_out, py_compute_queue, py_node_to_index,
                                                      py_node_to_dependencies, py_node_to_profile, py_storage_to_len,
                                                      py_options, profile, sentinel) for _ in range(computation_task)]
            logger.info(f"computation_task started")

            reader_tasks = [executor.submit(queue_event_reader, queue_in, stream_in, node_to_index, storage_to_len[INPUT_STORAGE])
                            for stream_in in inputs]
            logger.info(f"reader_tasks started")


            writer_tasks = [
                executor.submit(queue_event_writer, queue_out, stream_out, output_item_index, sentinel) for
                stream_out in outputs]
            logger.info(f"writer_tasks started")

            reader_result = [t.result() for t in reader_tasks]

            logger.info(f"reader_tasks finished")

            [queue_in.put(sentinel) for _ in range(computation_task)]
            computation_result = [ray.get(t) for t in compute_task]
            logger.info(f"computation_task finished")
            for _ in writer_tasks:
                queue_out.put(sentinel)

            writer_result = [t.result() for t in writer_tasks]
            logger.info(f"writer_tasks finished")
        except:
            queue_in.terminated = True
            queue_out.terminated = True
            raise


def run_synchronous(allocation_rule, static_path, files_in, files_out, **kwargs):
    compute_queue, node_to_index, node_to_dependencies, node_to_profile, output_item_index, storage_to_len, options, profile = load_financial_structure(
        allocation_rule, static_path)

    if files_in is None:
        stream_in = sys.stdin.buffer
    else:
        stream_in = open(files_in[0], 'rb')

    if files_out is not None:
        files_out = files_out[0]

    stream_type, len_sample = read_stream_header(stream_in)
    with EventWriter(files_out, output_item_index, len_sample + 4) as event_writer:
        for event_id, input_loss, input_not_null in read_event(stream_in, node_to_index, storage_to_len[INPUT_STORAGE], len_sample):
            output_loss, output_not_null = compute_event(compute_queue, node_to_index, node_to_dependencies,
                                                         node_to_profile, storage_to_len, options,
                                                         input_loss, input_not_null, profile)
            event_writer.write((event_id, output_loss, output_not_null))
