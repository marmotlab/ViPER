import ray
import numpy as np
import torch
import csv

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *


def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    checkpoint = torch.load(f'{model_path}/checkpoint.pth', weights_only=True, map_location=device)

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    dist_history = []
    max_dist_history = []
    explored_rate_history = []
    safe_rate_history = []
    success_rate_history = []
    all_length_history = []
    all_safe_rate_history = []
    all_explored_rate_history = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(dist_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                dist_history.append(metrics['travel_dist'])
                max_dist_history.append(metrics['max_travel_dist'])
                explored_rate_history.append(metrics['explored_rate'])
                safe_rate_history.append(metrics['safe_rate'])
                success_rate_history.append(metrics['success_rate'])
                all_length_history.extend(metrics['length_history'])
                all_safe_rate_history.extend(metrics['safe_rate_history'])
                all_explored_rate_history.extend(metrics['explored_rate_history'])

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        print('=====================================')
        print('| Test:', FOLDER_NAME)
        print('| Total test:', NUM_TEST)
        print('| Number of agents:', TEST_N_AGENTS)
        print('| Unbounded evader speed:', UNBOUND_SPEED)
        print('| Average max length:', np.array(max_dist_history).mean())
        print('| Std max length:', np.array(max_dist_history).std())
        print('| Average explored rate:', np.array(explored_rate_history).mean())
        print('| Average cleared rate:', np.array(safe_rate_history).mean())
        print('| Average success rate:', np.array(success_rate_history).mean())

        if SAVE_CSV:
            idx = np.array(all_length_history).argsort()
            all_length_history = np.array(all_length_history)[idx]
            all_safe_rate_history = np.array(all_safe_rate_history)[idx]
            all_explored_rate_history = np.array(all_explored_rate_history)[idx]
            with open(f'results/result_rl_n={TEST_N_AGENTS}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['length', 'safe', 'explore'])
                csv_data = np.concatenate([all_length_history.reshape(-1, 1), all_safe_rate_history.reshape(-1, 1),
                                           all_explored_rate_history.reshape(-1, 1)], axis=-1)
                writer.writerows(csv_data)
            print('CSV saved')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, device=self.device,
                            save_image=SAVE_GIFS, greedy=True, test=True)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()
