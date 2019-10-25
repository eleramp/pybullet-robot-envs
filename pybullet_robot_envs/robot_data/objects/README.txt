HOW TO create a convex decomposition using Hierarchical Approximate Convex Decomposition (v-HACD) of bullet3:

git clone https://github.com/bulletphysics/bullet3
cd bullet3/build3
./premake4_osx gmake
cd gmake
make test_vhacd
../../bin/test_vhacd_gmake_x64_release --input file.obj --output file_vhacd.obj --log log.txt
