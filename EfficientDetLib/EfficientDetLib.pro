#-------------------------------------------------
#
# Project created by QtCreator 2021-06-15T10:13:18
#
#-------------------------------------------------

QT       -= core gui

TARGET = EfficientDet
TEMPLATE = lib

DEFINES += DLDETEXTIONLIB_LIBRARY  \
        _GLIBCXX_USE_CXX11_ABI=0

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    src/dldetectionlib.cpp \
    src/model/efficientnet/mbconvblock.cpp \
    src/model/efficientnet/efficientnet.cpp \
    src/model/common/conv2dstaticsamepadding.cpp \
    src/model/utils.cpp \
    src/model/efficientdet/efficientdet.cpp \
    src/model/efficientdet/classifier.cpp \
    src/model/efficientdet/regressor.cpp \
    src/model/efficientdet/bifpn.cpp \
    src/model/efficientdet/separableconvblock.cpp \
    src/model/efficientdet/anchors.cpp \
    src/model/common/swish.cpp \
    src/model/common/memoryefficientswish.cpp \
    src/model/common/maxpool2dstaticsamepadding.cpp \
    src/model/efficientdet/efficientnetwrapper.cpp

HEADERS += \
    src/dldetectionlib.h \
    src/model/efficientnet/mbconvblock.h \
    src/model/efficientnet/efficientnet.h \
    src/model/efficientnet/efficientnetconfig.h \
    src/model/efficientnet/efficientnettype.h \
    src/model/common/conv2dstaticsamepadding.h \
    src/model/common/swishimplementation.h \
    src/model/utils.h \
    src/model/efficientdet/efficientdet.h \
    src/model/efficientdet/efficientdettype.h \
    src/model/efficientdet/classifier.h \
    src/model/efficientdet/regressor.h \
    src/model/efficientdet/bifpn.h \
    src/model/efficientdet/separableconvblock.h \
    src/model/efficientdet/anchors.h \
    src/model/common/swish.h \
    src/model/common/memoryefficientswish.h \
    src/model/common/maxpool2dstaticsamepadding.h \
    src/model/efficientdet/efficientnetwrapper.h \
    src/model/efficientdet/efficientdetconfig.h

DESTDIR = ./bin

OBJECTS_DIR = ./bin/temp

INCLUDEPATH += ../include \
               ../include/libtorch \
               ../include/libtorch/torch/csrc/api/include \
               ./src/ \
               ./src/model \
               ./src/model/common


LIBS += -L$$PWD/../lib/libtorch/ \
                -lc10 \
                -ltensorpipe_agent \
                -lcaffe2_detectron_ops \
                -ltorchbind_test \
                -lcaffe2_module_test_dynamic \
                -ltorch_cpu \
                -lcaffe2_observers \
                -ltorch_global_deps \
                -ljitbackend_test  \
                -ltorch_python\
                -lprocess_group_agent \
                -ltorch \
                -lshm \
        -L$$PWD/../lib/opencv/ \
                -lopencv_core \
                -lopencv_features2d \
                -lopencv_imgcodecs \
                -lopencv_imgproc \
                -lopencv_flann \

unix {
    target.path = /usr/lib
    INSTALLS += target
}
