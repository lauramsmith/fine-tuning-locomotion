"""SAC agent that yields after every environment episode."""

import collections
import numpy as np
import os
import tensorflow as tf
import time

import sac_dev.learning.sac_agent as sac_agent
import sac_dev.util.logger as logger
import sac_dev.util.mpi_util as mpi_util
import sac_dev.util.replay_buffer as replay_buffer

import pickle

class YieldingSACAgent(sac_agent.SACAgent):

    def __init__(self, *args, **kwargs):
        if mpi_util.get_num_procs() != 1:
            raise ValueError("YieldingSACAgent cannot run with multiprocessing")
        super().__init__(*args, **kwargs)

    def train(self, max_samples, test_episodes, output_dir, output_iters, variant=None):
        log_file = os.path.join(output_dir, "log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file, variant=variant)

        video_dir = os.path.join(output_dir, "videos")
        if (mpi_util.is_root_proc()):
            os.makedirs(video_dir, exist_ok=True)
            model_dir = os.path.join(output_dir, "train")
            os.makedirs(model_dir, exist_ok=True)
        self._tf_writer = tf.summary.FileWriter(
            os.path.join(output_dir, "tensorboard"), graph=self._sess.graph)

        iter = 0
        total_train_path_count = 0
        test_return = 0
        total_test_path_count = 0
        start_time = time.time()

        init_train_func = self._init_train()
        for _ in init_train_func:
          yield

        num_procs = 1
        local_samples_per_iter = int(np.ceil(self._samples_per_iter / num_procs))
        local_test_episodes = int(np.ceil(test_episodes / num_procs))

        total_samples = 0
        print("Training")

        while (total_samples < max_samples):
            update_normalizer = self._enable_normalizer_update(total_samples)
            rollout_train_func = self._rollout_train(local_samples_per_iter, update_normalizer)
            for ret in rollout_train_func:
              yield
            train_return, train_path_count, new_sample_count, metrics = ret
            train_return = mpi_util.reduce_mean(train_return)
            train_path_count = mpi_util.reduce_sum(train_path_count)
            new_sample_count = mpi_util.reduce_sum(new_sample_count)

            total_train_path_count += train_path_count

            total_samples = self.get_total_samples()
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours

            log_dict = {
                "Iteration": iter,
                "Wall_Time": wall_time,
                "Samples": total_samples,
                "Train_Return": train_return,
                "Train_Paths": total_train_path_count,
                "Test_Return": test_return,
                "Test_Paths": total_test_path_count}
            for metric_name, value in metrics.items():
                if metric_name == "max_torque":
                    log_dict["Max_Torque"] = mpi_util.reduce_max(value)
                    continue
                log_dict[metric_name] = mpi_util.reduce_mean(value)

            self._log(log_dict, iter)

            if (self._need_normalizer_update() and iter == 0):
                self._update_normalizers()

            self._update(iter, new_sample_count)

            if (self._need_normalizer_update()):
                self._update_normalizers()

            if (iter % output_iters == 0):
                rollout_test_func = self._rollout_test(local_test_episodes, print_info=False)
                for ret in rollout_test_func:
                  yield
                test_return, test_path_count = ret
                test_return = mpi_util.reduce_mean(test_return)
                total_test_path_count += mpi_util.reduce_sum(test_path_count)

                self._log({
                    "Test_Return": test_return,
                    "Test_Paths": total_test_path_count
                }, iter)

                if (mpi_util.is_root_proc()):
                    model_file = os.path.join(model_dir, f"model-{iter:06}.ckpt")
                    self.save_model(model_file)
                    buffer_file = os.path.join(model_dir, f"buffer.pkl")
                    file = open(buffer_file, "wb")
                    pickle.dump(self._replay_buffer, file)
                    file.close()

                self._logger.print_tabular()
                self._logger.dump_tabular()

            else:
                self._logger.print_tabular()

            iter += 1

        self._tf_writer.close()
        self._tf_writer = None
        return

    def _init_train(self):
        super(sac_agent.SACAgent, self)._init_train()
        num_procs = mpi_util.get_num_procs()
        local_init_samples = int(np.ceil(self._init_samples / num_procs))
        collect_func = self._collect_init_samples(local_init_samples)
        for _ in collect_func:
          yield

    def _collect_init_samples(self, max_samples):
        print("Collecting {} initial samples".format(max_samples))
        sample_count = 0
        next_benchmark = 1000
        update_normalizer = self._enable_normalizer_update(sample_count)
        start_time = time.time()

        while (sample_count < max_samples):
            rollout_func = self._rollout_train(1, update_normalizer)
            next(rollout_func)
            _, _, new_sample_count, _ = next(rollout_func)
            sample_count += new_sample_count
            print("samples: {}/{}".format(sample_count, max_samples))
            if sample_count >= next_benchmark:
                print("Collected {} initial samples in {} sec".format(
                    sample_count, time.time() - start_time))
                next_benchmark += 1000
            yield

        if (self._need_normalizer_update()):
            self._update_normalizers()

        yield sample_count

    def _rollout_train(self, num_samples, update_normalizer):
        new_sample_count = 0
        total_return = 0
        path_count = 0
        all_metrics = collections.defaultdict(list)

        while (new_sample_count < num_samples):
            path, _, metrics = self._rollout_path(test=False)
            yield
            path_id = self._replay_buffer.store(path)
            valid_path = path_id != replay_buffer.INVALID_IDX

            if not valid_path:
                assert False, "Invalid path detected"

            path_return = path.calc_return()

            if update_normalizer:
                self._record_normalizers(path)

            for metric_name in metrics:
                all_metrics[metric_name].append(metrics[metric_name][0])
            all_metrics["max_torque"].append(path.calc_max_torque())
            new_sample_count += path.pathlength()
            total_return += path_return
            path_count += 1

        avg_return = total_return / path_count
        metrics["max_torque"] = (None, np.max)
        aggregate_metrics = {}
        for metric_name, val_list in all_metrics.items():
            aggregate_fn = metrics[metric_name][1]
            aggregate_metrics[metric_name] = aggregate_fn(val_list)

        yield avg_return, path_count, new_sample_count, aggregate_metrics

    def _rollout_test(self, num_episodes, print_info=False, task_name=""):
        total_return = 0
        for e in range(num_episodes):
            path, _, _ = self._rollout_path(test=True)
            yield
            path_return = path.calc_return()
            total_return += path_return

            if (print_info):
                logger.Logger.print("Task: "+task_name)
                logger.Logger.print("Episode: {:d}".format(e))
                logger.Logger.print("Curr_Return: {:.3f}".format(path_return))
                logger.Logger.print("Avg_Return: {:.3f}\n".format(total_return / (e + 1)))
            
            yield

        avg_return = total_return / num_episodes
        yield avg_return, num_episodes
