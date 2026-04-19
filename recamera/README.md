# Steps to compile

This won't compile out of the box, first of all you need to follow the steps in the
reCamera documentation, specifically "Developpe sur reCamera avec c&cpp"
the directories such as _components_, _host-tools_, or _sg2002_recamera_emmc_ can all be found in the
repositories that must be cloned (given in step 1).

All you have to do then is to clone their contents inside of the specificed areas.

Once this is done,
from the root of this directory do this:
   1 export SG200X_SDK_PATH=$(pwd)/sg2002_recamera_emmc
   2 export PATH=$(pwd)/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
then
   3 cd build
   4 cmake ..
   5 make


Happy coding (or not)
