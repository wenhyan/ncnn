#include <iostream>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int main()
{
    ncnn::Net study_relu;

    if (study_relu.load_param("study_relu.param"))
        exit(-1);
    if (study_relu.load_model("study_relu.bin"))
        exit(-1);

    ncnn::Mat in_pad(256, 256, 3);
    ncnn::Mat output;

    ncnn::Extractor ex = study_relu.create_extractor();

    ex.input("input", in_pad);

    ex.extract("output", output);

    std::cout << "test\n";
    return 0;
}