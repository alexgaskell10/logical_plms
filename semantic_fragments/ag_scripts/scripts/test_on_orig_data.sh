## Script to test a fine-tuned model on the fragments data
top=..
root=../generate_challenge/brazil_fragments
for dir in $(ls $root | grep -v README.md)
do
    eval_dir=$root/$dir/test/

    python sen_pair_classification.py \
        --task_name ag_polarity \
        --bert_model bert-base-uncased  \
        --output_dir $top/_experiments/exp/$dir \
        --data_dir $eval_dir \
        --do_eval \
        --do_lower_case \
        --max_seq_length 128 \
        --run_existing $top/_experiments/propositional_v2/pytorch_model.bin \
        --bert_config $top/_experiments/propositional_v2/bert_config.json
done