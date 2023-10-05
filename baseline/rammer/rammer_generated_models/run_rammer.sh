cd /root/rammer_generated_models/bert
./main_test | tee bert.rammer.log 2>&1

cd /root/rammer_generated_models/resnext/
./main_test | tee resnext.rammer.log 2>&1

cd /root/rammer_generated_models/lstm/
./main_test | tee lstm.rammer.log 2>&1
