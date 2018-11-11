#include "TestData.h"
#include "Utils.h"

using namespace pu;

int main()
{

	cv::namedWindow("VerticalPlane", cv::WINDOW_NORMAL);
	cv::namedWindow("HorizontalPlane", cv::WINDOW_NORMAL);
	cv::namedWindow("ShearPlanes", cv::WINDOW_NORMAL);
	cv::namedWindow("SpiralShear", cv::WINDOW_NORMAL);
	cv::namedWindow("Peaks", cv::WINDOW_NORMAL);

	cv::Mat verticalplane = VerticalPlane();
	cv::Mat horizontalplane = HorizontalPlane();
	cv::Mat shearplane = ShearPlanes();
	cv::Mat spiralshear = SpiralShear();
	cv::Mat peaks = Peaks();

	cv::imshow("VerticalPlane", ToDisplayable(verticalplane));
	cv::imshow("HorizontalPlane", ToDisplayable(horizontalplane));
	cv::imshow("ShearPlanes", ToDisplayable(shearplane));
	cv::imshow("SpiralShear", ToDisplayable(spiralshear));
	cv::imshow("Peaks", ToDisplayable(peaks));

	cv::waitKey(0);

	pu::AddSaltPepperNoise(verticalplane);
	cv::imshow("VerticalPlane", ToDisplayable(verticalplane));

	cv::waitKey(0);
	cv::destroyAllWindows();
}