import numpy as np
import sys


def read_last_line_of_file(filename):
    try:
        with open(filename, 'r') as f:
            for line in f:
                if len(line) == 0:
                    continue
                last_line = line
        return last_line
    except:
        print(filename, sys.exc_info()[0])
        return None


def average(lists):
    max_size = max([len(l) for l in lists])
    avg_list = [0.0] * max_size
    sizes = [0] * max_size
    for i in range(max_size):
        for l in lists:
            try:
                avg_list[i] += l[i]
                sizes[i] += 1
            except Exception:
                pass
    avg_list = [avg_list[i] / sizes[i]
                for i in range(max_size) if sizes[i] != 0]
    return avg_list


def extract_main_exp():
    exclude_exp = False
    setting = sys.argv[1]
    dataset = sys.argv[2]
    exec_num = sys.argv[3]
    retrain = (sys.argv[4] == '1')
    test_domains = ['calendar', 'blocks', 'housing', 'restaurants',
                    'publications', 'recipes', 'socialnetwork', 'basketball']
    if exclude_exp:
        test_domains = ['exclude_' + domain for domain in test_domains]
    test_log_filename = 'test.log'
    test_logs = []
    for domain in test_domains:
        model_dir = ('execs/' + setting + '/' + dataset + '/' +
                     domain + '/exec' + str(exec_num))
        if retrain:
            model_dir = model_dir + '/retrain'
        test_logs.append(model_dir + '/' + test_log_filename)
    accuracies = []
    domains_with_result = ['average']
    for id, log in enumerate(test_logs):
        domain = test_domains[id]
        last_line = read_last_line_of_file(log)
        if last_line and 'mean acc' in last_line:
            accuracy = float(last_line.split(' ')[13].strip(';'))
            accuracies.append(accuracy)
            domains_with_result.append(domain)
    accuracies = [np.mean(accuracies)] + accuracies
    return domains_with_result, accuracies


def extract_repeated_run_exp():
    setting = sys.argv[1]
    dataset = sys.argv[2]
    domain = sys.argv[3]
    exec_num_st, exec_num_ed = [int(exec_num)
                                for exec_num in sys.argv[4].split(',')]
    accuracies = []
    mrrs = []
    hits = []
    for exec_num in range(exec_num_st, exec_num_ed + 1):
        test_log = 'execs/' + setting + '/' + dataset + '/' + \
            domain + '/exec' + str(exec_num) + '/test.log'
        last_line = read_last_line_of_file(test_log)
        assert(last_line and 'mean acc' in last_line)
        accuracy = float(last_line.split(' ')[13].strip(';'))
        mrr = float(last_line.split(' ')[17].strip('.'))
        hit = float(last_line.split(' ')[20][:6])
        accuracies.append(accuracy)
        mrrs.append(mrr)
        hits.append(hit)
    print('acc: %s' % ' '.join([str(acc) for acc in accuracies]))
    print('mrr: %s' % ' '.join([str(mrr) for mrr in mrrs]))
    print('hit: %s' % ' '.join([str(hit) for hit in hits]))
    import numpy as np
    print('mean acc: %.4f, std: %.4f; mean mrr: %.4f, std: %.4f;'
          'mean hit: %.4f, std: %.4f.' %
          (np.mean(accuracies), np.std(accuracies),
           np.mean(mrrs), np.std(mrrs), np.mean(hits), np.std(hits)))


def extract_downsample_exp():
    setting = sys.argv[1]
    dataset = sys.argv[2]
    exec_str = sys.argv[3]
    exec_start, exec_end = [int(exec_num) for exec_num in exec_str.split(',')]
    exec_nums = range(exec_start, exec_end)
    retrain = (sys.argv[4] == '1')
    init_type = None
    if len(sys.argv) > 5:
        init_type = sys.argv[5]
    test_domains = ['calendar', 'blocks', 'housing', 'restaurants',
                    'publications', 'recipes', 'socialnetwork', 'basketball']
    # test_domains = ['calendar', 'blocks', 'housing', 'restaurants',
    #                 'publications', 'recipes', 'socialnetwork']
    avg_accuracies = []
    for domain in test_domains:
        avg_accuracy = extract_downsample_exp_domain(setting,
                                                     dataset,
                                                     domain,
                                                     exec_nums,
                                                     retrain,
                                                     init_type)
        avg_accuracies.append(avg_accuracy)
    avg_accuracies.append(average(avg_accuracies))
    test_domains.append('average')
    for i, acc in enumerate(avg_accuracies):
        print(test_domains[i],
              ' '.join([str(acc_single) for acc_single in acc]))


def extract_downsample_exp_domain(setting, dataset, domain, exec_nums,
                                  retrain, init_type=None):
    test_log_filename = 'test.log'
    downsample_ratios = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    all_accuracies = []
    for exec_num in exec_nums:
        accuracies = []
        for ratio in downsample_ratios:
            model_dir = ('execs/' + setting + '/' + dataset + '/' +
                         domain + '/sample_' + str(ratio) +
                         '/exec' + str(exec_num))
            if init_type:
                model_dir = model_dir + '/' + init_type
            if retrain:
                model_dir = model_dir + '/retrain'
            log = model_dir + '/' + test_log_filename
            last_line = read_last_line_of_file(log)
            if last_line and 'mean acc' in last_line:
                accuracy = float(last_line.split(' ')[13].strip(';'))
                accuracies.append(accuracy)
        all_accuracies.append(accuracies)
    # all_accuracies.append(np.mean(all_accuracies, 0))
    # return all_accuracies
    return average(all_accuracies)


if __name__ == '__main__':
    domains_with_result, accuracies = extract_main_exp()
    print(' '.join([d for d in domains_with_result]))
    print(' '.join([str(acc) for acc in accuracies]))

    # extract_downsample_exp()

    # extract_repeated_run_exp()
