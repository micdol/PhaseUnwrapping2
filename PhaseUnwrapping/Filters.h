#pragma once
#include <opencv2\opencv.hpp>

namespace pu
{
	namespace filters
	{
		cv::Mat MeanPhaseFilter(const cv::Mat& wrapped, int k);

		cv::Mat MedianPhaseFilter(const cv::Mat& wrapped, int k);
	}
}