cd lstm/build
./main_test > ../../../artifacts/figure5/logs/lstm.rammer.log 2>&1
cd ../../seq2seq/build
./main_test > ../../../artifacts/figure5/logs/seq2seq.rammer.log 2>&1
cd ../../deepspeech2/build
./main_test > ../../../artifacts/figure5/logs/deepspeech2.rammer.log 2>&1
cd ../../bert
./main_test > ../../artifacts/figure5/logs/bert.rammer.log 2>&1
cd ../resnext/build
./main_test > ../../../artifacts/figure5/logs/resnext.rammer.log 2>&1
cd ../../nasnet/build
./main_test > ../../../artifacts/figure5/logs/nasnet.rammer.log 2>&1
cd ../..
