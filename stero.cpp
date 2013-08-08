#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

int main(int argc, char *argv[]) {
    using namespace cv;

    if (argc != 3) {
        std::cout << "usage ./stero [left.png] [right.png]" << std::endl;
        return 1;
    }

    Mat left = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat right = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if (!left.data || !right.data) {
        std::cout << "failed opening/reading images" << std::endl;
        return 2;
    }

    return 0;
}
