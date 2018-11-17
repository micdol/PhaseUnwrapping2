#include "TestData.h"
#include "Utils.h"
#include "QualityMaps.h"
#include "Wrappers.h"
#include "Filters.h"

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

	verticalplane = Wrap(verticalplane);
	horizontalplane = Wrap(horizontalplane);
	shearplane = Wrap(shearplane);
	spiralshear = Wrap(spiralshear);
	peaks = Wrap(peaks);

	cv::imshow("VerticalPlane", verticalplane);
	cv::imshow("HorizontalPlane", horizontalplane);
	cv::imshow("ShearPlanes", shearplane);
	cv::imshow("SpiralShear", spiralshear);
	cv::imshow("Peaks", peaks);
	cv::waitKey(0);
	/*
	int k = 5;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::PDV(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::PDV(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::PDV(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::PDV(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::PDV(peaks, k)));
	cv::waitKey(0);

	k = 9;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::PDV(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::PDV(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::PDV(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::PDV(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::PDV(peaks, k)));
	cv::waitKey(0);

	k = 5;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::MaxAbsGrad(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::MaxAbsGrad(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::MaxAbsGrad(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::MaxAbsGrad(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::MaxAbsGrad(peaks, k)));
	cv::waitKey(0);

	k = 9;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::MaxAbsGrad(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::MaxAbsGrad(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::MaxAbsGrad(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::MaxAbsGrad(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::MaxAbsGrad(peaks, k)));
	cv::waitKey(0);

	AddSaltPepperNoise(verticalplane);
	AddSaltPepperNoise(horizontalplane);
	AddSaltPepperNoise(shearplane);
	AddSaltPepperNoise(spiralshear);
	AddSaltPepperNoise(peaks);
	cv::imshow("VerticalPlane", ToDisplayable(verticalplane));
	cv::imshow("HorizontalPlane", ToDisplayable(horizontalplane));
	cv::imshow("ShearPlanes", ToDisplayable(shearplane));
	cv::imshow("SpiralShear", ToDisplayable(spiralshear));
	cv::imshow("Peaks", ToDisplayable(peaks));
	cv::waitKey(0);
	
	k = 5;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::PDV(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::PDV(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::PDV(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::PDV(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::PDV(peaks, k)));
	cv::waitKey(0);

	k = 9;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::PDV(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::PDV(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::PDV(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::PDV(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::PDV(peaks, k)));
	cv::waitKey(0);

	k = 5;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::MaxAbsGrad(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::MaxAbsGrad(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::MaxAbsGrad(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::MaxAbsGrad(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::MaxAbsGrad(peaks, k)));
	cv::waitKey(0);

	k = 9;
	cv::imshow("VerticalPlane", ToDisplayable(quality_maps::MaxAbsGrad(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(quality_maps::MaxAbsGrad(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(quality_maps::MaxAbsGrad(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(quality_maps::MaxAbsGrad(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(quality_maps::MaxAbsGrad(peaks, k)));
	cv::waitKey(0);
	*/
	int k = 3;
	cv::imshow("VerticalPlane", ToDisplayable(filters::MeanPhaseFilter(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(filters::MeanPhaseFilter(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(filters::MeanPhaseFilter(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(filters::MeanPhaseFilter(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(filters::MeanPhaseFilter(peaks, k)));
	cv::waitKey(0);

	k = 5;
	cv::imshow("VerticalPlane", ToDisplayable(filters::MeanPhaseFilter(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(filters::MeanPhaseFilter(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(filters::MeanPhaseFilter(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(filters::MeanPhaseFilter(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(filters::MeanPhaseFilter(peaks, k)));
	cv::waitKey(0);

	k = 9;
	cv::imshow("VerticalPlane", ToDisplayable(filters::MeanPhaseFilter(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(filters::MeanPhaseFilter(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(filters::MeanPhaseFilter(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(filters::MeanPhaseFilter(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(filters::MeanPhaseFilter(peaks, k)));
	cv::waitKey(0);

	k = 3;
	cv::imshow("VerticalPlane", ToDisplayable(filters::MedianPhaseFilter(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(filters::MedianPhaseFilter(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(filters::MedianPhaseFilter(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(filters::MedianPhaseFilter(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(filters::MedianPhaseFilter(peaks, k)));
	cv::waitKey(0);

	k = 5;
	cv::imshow("VerticalPlane", ToDisplayable(filters::MedianPhaseFilter(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(filters::MedianPhaseFilter(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(filters::MedianPhaseFilter(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(filters::MedianPhaseFilter(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(filters::MedianPhaseFilter(peaks, k)));
	cv::waitKey(0);

	k = 9;
	cv::imshow("VerticalPlane", ToDisplayable(filters::MedianPhaseFilter(verticalplane, k)));
	cv::imshow("HorizontalPlane", ToDisplayable(filters::MedianPhaseFilter(horizontalplane, k)));
	cv::imshow("ShearPlanes", ToDisplayable(filters::MedianPhaseFilter(shearplane, k)));
	cv::imshow("SpiralShear", ToDisplayable(filters::MedianPhaseFilter(spiralshear, k)));
	cv::imshow("Peaks", ToDisplayable(filters::MedianPhaseFilter(peaks, k)));
	cv::waitKey(0);

	cv::destroyAllWindows();
}