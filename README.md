# opencv_face_reid

#Install OpenVINO

https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino


#Build OpenCV4 on Ubuntu16.04

https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/

#Build OpenCV with OpenVINO support

https://github.com/opencv/opencv/wiki/Intel%27s-Deep-Learning-Inference-Engine-backend#linux

source /opt/intel/openvino/bin/setupvars.sh
export ngraph_DIR=/opt/intel/openvino/deployment_tools/ngraph/cmake/
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion" \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D
OPENCV_PYTHON3_INSTALL_PATH=$cwd/OpenCV-$cvVersion-py3/lib/python3.5/site-packages
\
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -DWITH_INF_ENGINE=ON \
    -DENABLE_CXX11=ON \
      -D BUILD_EXAMPLES=ON ..

#Run demo
xpython ./faceid.py -d models/fd -e models/fr/face-reidentification-retail-0095 -f models/fas/ feathernetB -m img/ir_small.png -p img/depth_revert.png
