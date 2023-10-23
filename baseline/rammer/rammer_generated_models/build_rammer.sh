cd /root/rammer_generated_models/bert
rm -r CMakeFiles CMakeCache.txt Makefile
cmake . && make -j > rammer.build.log 2>&1

cd /root/rammer_generated_models/resnext/
rm -r CMakeFiles CMakeCache.txt Makefile
cmake . && make -j > rammer.build.log 2>&1

cd /root/rammer_generated_models/lstm/
rm -r CMakeFiles CMakeCache.txt Makefile
cmake . && make -j > rammer.build.log 2>&1
