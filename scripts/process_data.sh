for domain in calendar blocks housing restaurants publications recipes socialnetwork basketball
do
  python -m scripts/process_data overnight data/overnight/${domain} ${domain}.paraphrases.train.examples train
  python -m scripts/process_data overnight data/overnight/${domain} ${domain}.paraphrases.test.examples test
  python -m scripts/process_data overnight data/overnight/${domain} ${domain}.result candidate --overnight_train_exmp_file ${domain}.paraphrases.train.examples --overnight_test_exmp_file ${domain}.paraphrases.test.examples
done
