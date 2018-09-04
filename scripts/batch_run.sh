gpu=$1
train_script=$2
dataset=$3
exec_num=$4

for domain in calendar blocks housing restaurants publications recipes socialnetwork basketball
do
  # train with no pretraining and then test
  ./scripts/commands.sh $train_script $gpu in-domain $dataset $domain $exec_num
  ./scripts/commands.sh test $gpu in-domain $dataset $domain $exec_num

  # retrain with full training set and then test
  ./scripts/commands.sh train_grid_retrain $gpu in-domain $dataset $domain $exec_num
  ./scripts/commands.sh test_retrain $gpu in-domain $dataset $domain $exec_num

  # pretraining
  ./scripts/commands.sh $train_script $gpu cross-domain $dataset exclude_$domain $exec_num
  ./scripts/commands.sh test $gpu cross-domain $dataset $domain $exec_num

  # train with pretraining and then test
  ./scripts/commands.sh $train_script $gpu cross-domain $dataset $domain $exec_num exclude_${domain} $exec_num
  ./scripts/commands.sh test $gpu cross-domain $dataset $domain $exec_num
  
  # retrain with full training set and then test
  ./scripts/commands.sh train_grid_retrain $gpu cross-domain $dataset $domain $exec_num
  ./scripts/commands.sh test_retrain $gpu cross-domain $dataset $domain $exec_num

done
