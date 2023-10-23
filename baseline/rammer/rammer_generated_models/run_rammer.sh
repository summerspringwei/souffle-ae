cd /root/rammer_generated_models/bert
./main_test | tee bert.rammer.log 2>&1
RAMMER_BERT_LATENCY=$(cat bert.rammer.log | grep "Summary" | awk -F'[][]' '{print $4}' | awk '{ print $3 }')

cd /root/rammer_generated_models/resnext/
./main_test | tee resnext.rammer.log 2>&1
RAMMER_RESNEXT_LATENCY=$(cat resnext.rammer.log | grep "Summary" | awk -F'[][]' '{print $4}' | awk '{ print $3 }')
NUM_TIMES=3 # This is a resnext with is 3 times smaller than resnext-101
RAMMER_RESNEXT_LATENCY=$(python3 -c "print(${RAMMER_RESNEXT_LATENCY} * ${NUM_TIMES})")

cd /root/rammer_generated_models/lstm/
./main_test | tee lstm.rammer.log 2>&1
RAMMER_LSTM_LATENCY=$(cat lstm.rammer.log | grep "Summary" | awk -F'[][]' '{print $4}' | awk '{ print $3 }')

cd /root/rammer_generated_models/

echo "Rammer:," ${RAMMER_BERT_LATENCY}, ${RAMMER_RESNEXT_LATENCY},\
 ${RAMMER_LSTM_LATENCY}, "failed", "failed", "failed" > table3_rammer.csv