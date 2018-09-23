import os
import argparse
import jsonpickle
import random
import json

from src import utils
from src import nlp_service


def save_to_file(out_path, data):
    """save processed data to file

    Args:
        out_path: the destination path
        data: array-like, each element is an example (a list of tokens)
    """
    with open(out_path, 'w') as f:
        for example in data:
            f.write(utils.format_list(example) + "\n")


def save_dict_to_file(out_path, data):
    """save a dictionary to a json file

    Args:
        out_path: the destination path
        data: a dictionary
    """
    with open(out_path, 'wb') as f_out:
        json.dump(data,
                  f_out,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))


def process_data(dataset, data_dir, in_file, mode, split_ratio,
                 overnight_train_exmp_file=None,
                 overnight_test_exmp_file=None):
    """Prepare data for semantic parsing

    source file and target file will be formatted such that they are aligned
    line by line. Each line of the source file is an input utterance, while
    each line of the target file is the corresponding canonical utterance

    Args:
        dataset: for now it's just 'overnight'. More in the future.
        data_dir: the directory of the data file
        in_file: the data file
        mode: {"train", "test", "candidate"}.
            We randomly split the training data into a training set
            and a validation set
        split_ratio: the split ratio of training data
    """
    in_path = os.path.join(data_dir, in_file)
    source_train_path = os.path.join(data_dir, "source.train")
    source_valid_path = os.path.join(data_dir, "source.valid")
    source_test_path = os.path.join(data_dir, "source.test")
    target_train_path = os.path.join(data_dir, "target.train")
    target_valid_path = os.path.join(data_dir, "target.valid")
    target_test_path = os.path.join(data_dir, "target.test")
    candidate_path = os.path.join(data_dir, "candidates.json")

    # read data
    if mode == "candidate":
        if dataset == "overnight":
            overnight_train_exmp_path = os.path.join(data_dir,
                                                     overnight_train_exmp_file)
            overnight_test_exmp_path = os.path.join(data_dir,
                                                    overnight_test_exmp_file)
            data = extract_candidates_overnight(in_path,
                                                overnight_train_exmp_path,
                                                overnight_test_exmp_path)
    else:
        if dataset == "overnight":
            data = process_data_overnight(in_path)

    # write to files
    if mode == "train":
        n = len(data)
        # split into training and validation
        n_valid = int(n * split_ratio)
        all_indices = range(n)
        valid_indices = random.sample(all_indices, n_valid)
        train_indices = [index for index in range(n)
                         if index not in valid_indices]
        source_train_set = [data[index][0] for index in train_indices]
        target_train_set = [data[index][1] for index in train_indices]
        source_valid_set = [data[index][0] for index in valid_indices]
        target_valid_set = [data[index][1] for index in valid_indices]
        save_to_file(source_train_path, source_train_set)
        save_to_file(target_train_path, target_train_set)
        save_to_file(source_valid_path, source_valid_set)
        save_to_file(target_valid_path, target_valid_set)
    elif mode == "test":
        source_test_set = [exp[0] for exp in data]
        target_test_set = [exp[1] for exp in data]
        save_to_file(source_test_path, source_test_set)
        save_to_file(target_test_path, target_test_set)
    elif mode == "candidate":
        save_dict_to_file(candidate_path, data)


def process_data_overnight(in_path):
    """Read and process LISP-formatted overnight data"""
    data = []
    with open(in_path, "r") as f_in:
        input_utter = None
        canonical_utter = None
        for line in f_in:
            line = line.strip()
            if line.startswith("(example"):
                if input_utter and canonical_utter:
                    data.append((input_utter, canonical_utter))
                input_utter = None
                canonical_utter = None
            elif line.startswith("(utterance"):
                input_utter = line[line.find('"') + 1:line.rfind('"')]
                input_utter = nlp_service.tokenize(input_utter)
            elif line.startswith("(original"):
                canonical_utter = line[line.find('"') + 1:line.rfind('"')]
                canonical_utter = nlp_service.tokenize(canonical_utter)
        # the last example
        if input_utter and canonical_utter:
            data.append((input_utter, canonical_utter))
        else:
            raise ValueError("something went wrong with the last example.")
    return data


def extract_candidates_overnight(in_path,
                                 overnight_train_exmp_path,
                                 overnight_test_exmp_path):
    """Extract candidates for web api data

    A typical snippet in the Sempre result file:

    Example: what is the start time for the weekly stand up meeting {
        Tokens: [what, is, the, start, time, for, the, weekly, stand, up,
            meeting]
        Lemmatized tokens: [what, be, the, start, time, for, the, weekly,
            stand, up, meeting]
        POS tags: [WP, VBD-AUX, DT, NN, NN, IN, DT, JJ, NN, IN, NN]
        NER tags: [O, O, O, O, O, O, O, SET, O, O, O]
        NER values: [null, null, null, null, null, null, null, null, null,
            null, null]
        targetFormula: (call edu.stanford.nlp.sempre.overnight.SimpleWorld.listValue (call edu.stanford.nlp.sempre.overnight.SimpleWorld.getProperty en.meeting.weekly_standup (string start_time)))
        targetValue: (list (time 13 0))
        Dependency children: [[], [attr->0, nsubj->4], [], [],
            [det->2, nn->3, prep_for->8], [], [], [],
            [det->6, amod->7, prep_up->10], [], []]
    }

    Args:
        in_path: the result file from the overnight paper used to extract
            the denotation of candidate canonical utterance/logical form.
            It is necessary to base the evaluation on denotations (as
            employed in the original paper) instead of directly on
            logical forms because there are many logical forms in the
            overnight data that have exactly the same denotation.
            For example, two logical forms might involve the same set of facts but are presented in a different order.
        overnight_train_exmp_path: path to the training examples.
        overnight_test_exmp_path: path to the testing examples.

    """
    # read training/testing example files to build connection between
    # logical form (lm) and canonical utterance
    lm2canonical = {}
    for exmp_path in [overnight_train_exmp_path, overnight_test_exmp_path]:
        with open(exmp_path, "r") as f_in:
            lm = None
            canonical_utter = None
            for line in f_in:
                if len(line) == 0:
                    continue
                line = line.strip()
                if line.startswith("(example"):
                    if lm and canonical_utter:
                        if lm in lm2canonical:
                            assert(lm2canonical[lm] == canonical_utter)
                        lm2canonical[lm] = canonical_utter
                    lm = None
                    canonical_utter = None
                elif line.startswith("(original"):
                    canonical_utter = line[line.find('"') + 1:line.rfind('"')]
                elif line.startswith("(call"):
                    lm = line
            # the last example
            if lm and canonical_utter:
                if lm in lm2canonical:
                    assert(lm2canonical[lm] == canonical_utter)
                lm2canonical[lm] = canonical_utter
            else:
                raise ValueError("something went wrong with the last example.")

    # read the result file from the overnight paper to build connection between
    # logical form and denotation
    lm2denotation = {}
    n_empty_denotation = 0
    in_example = False
    with open(in_path, "r") as f:
        for line in f:
            if len(line) == 0:
                continue
            line = line.strip()
            if line.startswith('Example: '):
                # input_utter = line[line.find(':') + 1:line.rfind('{')]
                # input_utter = input_utter.strip()
                in_example = True
                denotation = None
            elif in_example and line.startswith('targetFormula:'):
                assert('denotation' not in locals() or denotation is None)
                lm = line[15:]
            elif in_example and line.startswith('targetValue:'):
                assert('lm' in locals() and lm is not None)
                denotation_str = line[19:line.rfind(')')]
                # Not sure about the reason, but some logical forms in
                # the socialnetwork domain have empty denotation
                if len(denotation_str) == 0:
                    denotation = []
                    n_empty_denotation += 1
                    print('empty denotation: %s' % lm)
                else:
                    denotation = denotation_str.split(') (')
                    denotation = [v.strip('() ') for v in denotation]
                if lm in lm2denotation:
                    # if we've seen lm before, check denotation consistency
                    assert(set(lm2denotation[lm]) == set(denotation))
                lm2denotation[lm] = denotation
            elif in_example and line.startswith('}'):
                lm = None
                denotation = None
                in_example = False
    # assertion failed for socialnetwork
    # Not sure about the reason, but 20 examples in socialnetwork
    # domain are not used in the Sempre run, thus no denotation
    # assert(set(lm2canonical.keys()) == set(lm2denotation.keys()))

    # build connection between (candidate) canonical utterance and denotation
    candidates = []
    n_missing_lm = 0
    # for lm in lm2canonical:
    for lm in lm2canonical:
        canonical_utter = lm2canonical[lm]
        canonical_utter = nlp_service.tokenize(canonical_utter)
        canonical_utter = utils.format_list(canonical_utter)
        if lm in lm2denotation:
            denotation = lm2denotation[lm]
        else:
            print('denotation not found: %s' % lm)
            n_missing_lm = 0
            denotation = []
        candidates.append({'canonical_utterance': canonical_utter,
                           'denotation': denotation})
    missing_denotation = [cand for cand in candidates
                          if cand['denotation'] == []]
    n_missing_denotation = len(missing_denotation)
    n_cand = float(len(candidates))
    print('empty denotation: %d/%d=%f' %
          (n_empty_denotation, n_cand,
           n_empty_denotation / n_cand))
    print('missing logical form in result file: %d/%d=%f' %
          (n_missing_lm, n_cand, n_missing_lm / n_cand))
    print('overall no denotation: %d/%d=%f' %
          (n_missing_denotation, n_cand,
           n_missing_denotation / n_cand))
    return candidates


def extract_candidates_api(in_path):
    """Extract candidates for web api data"""
    candidates = []
    with open(in_path, "r") as f:
        for line in f:
            canonical_utter = line.strip()
            canonical_utter = nlp_service.tokenize(canonical_utter)
            canonical_utter = utils.format_list(canonical_utter)
            candidates.append({'canonical_utterance': canonical_utter,
                               'denotation': []})
    return candidates


def run():
    parser = argparse.ArgumentParser(description="process data from multiple"
                                     "domains for semantic parsing")
    parser.add_argument("domain", choices=["web_api", "overnight"],
                        help="domain")
    parser.add_argument("data_dir", help="the data directory")
    parser.add_argument("in_file", help="the original data file")
    parser.add_argument("mode", choices=["train", "test", "candidate"],
                        help="data processing purpose")
    parser.add_argument("--split_ratio", type=float, default=0.2,
                        help="ratio of training data for validation")
    parser.add_argument("--overnight_train_exmp_file",
                        help="file of the training examples of overnight data "
                        "used in generating candidates")
    parser.add_argument("--overnight_test_exmp_file",
                        help="file of the testing examples of overnight data "
                        "used in generating candidates")
    args = parser.parse_args()
    process_data(args.domain,
                 args.data_dir,
                 args.in_file,
                 args.mode,
                 args.split_ratio,
                 args.overnight_train_exmp_file,
                 args.overnight_test_exmp_file)


if __name__ == "__main__":
    run()
