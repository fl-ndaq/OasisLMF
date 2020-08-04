from .financial_structure import INPUT_STORAGE, TEMP_STORAGE, OUTPUT_STORAGE, STORE_LOSS_SUM_OPTION,\
    PROFILE, IL_PER_GUL, IL_PER_SUB_IL, PROPORTION, COPY, node_type, storage_type
from .policy import calc
from .common import float_equal_precision, nb_oasis_float, nb_oasis_int, np_oasis_float
from .queue import QueueTerminated

from numba import njit, boolean, typeof
import numpy as np
import logging
logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, error_model="numpy")
def compute_event(compute_queue, node_to_index, node_to_dependencies, node_to_profile, storage_to_len, options,
                  input_loss, input_not_null, profile):
    len_sample = input_loss.shape[1]
    temp_loss = np.zeros((storage_to_len[TEMP_STORAGE], len_sample), dtype=np_oasis_float)
    temp_not_null = np.zeros((storage_to_len[TEMP_STORAGE], ), dtype=boolean)
    output_loss = np.zeros((storage_to_len[OUTPUT_STORAGE], len_sample), dtype=np_oasis_float)
    output_not_null = np.zeros((storage_to_len[OUTPUT_STORAGE], ), dtype=boolean)
    losses = [input_loss,
              temp_loss,
              output_loss
              ]
    not_null = [input_not_null,
                temp_not_null,
                output_not_null
                ]

    if options[STORE_LOSS_SUM_OPTION]:
        losses_sum = np.zeros_like(temp_loss, dtype=np_oasis_float)
    else:
        losses_sum = temp_loss

    deductibles = np.zeros_like(temp_loss, dtype=np_oasis_float)
    over_limit = np.zeros_like(temp_loss, dtype=np_oasis_float)
    under_limit = np.zeros_like(temp_loss, dtype=np_oasis_float)

    for node in compute_queue:
        node_storage, node_index = node_to_index[node]
        if node[3] == PROFILE:
            is_not_null = False
            loss_sum = losses_sum[node_index]
            for dependency_storage, dependency_index in node_to_dependencies[node]:
                if not_null[dependency_storage][dependency_index]:
                    loss_sum += losses[dependency_storage][dependency_index]
                    if dependency_storage == TEMP_STORAGE:
                        deductibles[node_index] += deductibles[dependency_index]
                        over_limit[node_index] += over_limit[dependency_index]
                        under_limit[node_index] += under_limit[dependency_index]
                    is_not_null = True

            not_null[node_storage][node_index] = is_not_null
            if is_not_null:
                calc(profile[node_to_profile[node]],
                     losses[node_storage][node_index],
                     loss_sum,
                     deductibles[node_index],
                     over_limit[node_index],
                     under_limit[node_index])
                not_null[node_storage][node_index] = True

        elif node[3] == IL_PER_GUL:
            dependencies = node_to_dependencies[node]
            top_node_storage, top_node_index = dependencies[0]
            if not_null[top_node_storage][top_node_index]:
                node_loss = losses[node_storage][node_index]

                top_loss = losses[top_node_storage][top_node_index]
                for dependency_node_storage, dependency_node_index in dependencies[1:]:
                    node_loss += losses[dependency_node_storage][dependency_node_index]

                for i in range(top_loss.shape[0]):
                    if top_loss[i] < float_equal_precision:
                        node_loss[i] = 0
                    else:
                        node_loss[i] = top_loss[i] / node_loss[i]
                not_null[node_storage][node_index] = True

        elif node[3] == IL_PER_SUB_IL:
            (ba_node_storage, ba_node_index), (il_node_storage, il_node_index) = node_to_dependencies[node]
            if not_null[ba_node_storage][ba_node_index]:

                node_loss = losses[node_storage][node_index]
                ba_loss = losses[ba_node_storage][ba_node_index]
                if il_node_storage == TEMP_STORAGE:
                    il_loss = losses_sum[il_node_index]
                else:
                    il_loss = losses[il_node_storage][il_node_index]

                for i in range(node_loss.shape[0]):
                    if ba_loss[i] < float_equal_precision:
                        node_loss[i] = 0
                    else:
                        node_loss[i] = ba_loss[i] / il_loss[i]

                not_null[node_storage][node_index] = True

        elif node[3] == PROPORTION:
            (top_node_storage, top_node_index), (il_node_storage, il_node_index) = node_to_dependencies[node]

            if not_null[top_node_storage][top_node_index]:
                losses[node_storage][node_index] = losses[top_node_storage][top_node_index] * losses[il_node_storage][il_node_index]
                not_null[node_storage][node_index] = True

        elif node[3] == COPY:
            copy_node_storage, copy_node_index  = node_to_dependencies[node][0]
            if not_null[copy_node_storage][copy_node_index]:
                losses[node_storage][node_index] = losses[copy_node_storage][copy_node_index]
                not_null[node_storage][node_index] = True

    return output_loss, output_not_null


def event_computer(queue_in, queue_out, compute_queue, node_to_index, node_to_dependencies, node_to_profile,
                   storage_to_len, options, profile, sentinel):
    try:
        while True:
            event_in = queue_in.get()
            if event_in == sentinel:
                break

            event_id, input_loss, input_not_null = event_in

            logger.debug(f"computing {event_id}")
            input_loss = np.array(input_loss)
            input_not_null = np.array(input_not_null)
            output_loss, output_not_null = compute_event(compute_queue, node_to_index, node_to_dependencies, node_to_profile,
                                                         storage_to_len, options, input_loss, input_not_null, profile)
            logger.debug(f"computed {event_id}")

            try:
                queue_out.put((event_id, output_loss, output_not_null))
            except QueueTerminated:
                logger.warning(f"stopped because exception was raised")
                break

        logger.info(f"compute done")
    except Exception:
        logger.exception(f"Exception in compute")
        logger.error(input_loss)
        queue_in.terminated = True
        queue_out.terminated = True
        raise


try:
    import ray
except ImportError:
    pass
else:
    from numba.typed import List, Dict


    def numba_to_python(nb_compute_queue, nb_node_to_index, nb_node_to_dependencies, nb_node_to_profile,
                        nb_storage_to_len, nb_options):
        py_node_to_dependencies = {}
        for key, value in nb_node_to_dependencies.items():
            py_node_to_dependencies[key] = list(value)

        return list(nb_compute_queue), dict(nb_node_to_index), py_node_to_dependencies, dict(nb_node_to_profile), dict(
            nb_storage_to_len), dict(nb_options)

    def python_to_numba(py_compute_queue, py_node_to_index, py_node_to_dependencies, py_node_to_profile,
                        py_storage_to_len, py_options):
        nb_compute_queue = List.empty_list(node_type)
        for elm in py_compute_queue:
            nb_compute_queue.append(elm)

        nb_node_to_index = Dict.empty(node_type, storage_type)
        for key, val in py_node_to_index.items():
            nb_node_to_index[key] = val

        nb_node_to_dependencies = Dict.empty(key_type=node_type, value_type=typeof(List.empty_list(node_type)))
        for key, vals in py_node_to_dependencies.items():
            nb_vals = List.empty_list(node_type)
            for val in vals:
                nb_vals.append(val)
            nb_node_to_dependencies[key] = nb_vals

        nb_node_to_profile = Dict.empty(node_type, nb_oasis_int)
        for key, val in py_node_to_profile.items():
            nb_node_to_profile[key] = val

        nb_storage_to_len = Dict()
        for key, val in py_storage_to_len.items():
            nb_storage_to_len[key] = val

        nb_options = Dict()
        for key, val in py_options.items():
            nb_options[key] = val

        return nb_compute_queue, nb_node_to_index, nb_node_to_dependencies, nb_node_to_profile, nb_storage_to_len, nb_options

    @ray.remote
    def ray_event_computer(queue_in, queue_out, compute_queue, node_to_index, node_to_dependencies, node_to_profile,
                           storage_to_len, options, profile, sentinel):

        nb_obj = python_to_numba(compute_queue, node_to_index, node_to_dependencies, node_to_profile, storage_to_len, options)
        compute_queue, node_to_index, node_to_dependencies, node_to_profile, storage_to_len, options = nb_obj
        event_computer(queue_in, queue_out, compute_queue, node_to_index, node_to_dependencies, node_to_profile,
                       storage_to_len, options, profile, sentinel)
        return





