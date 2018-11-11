#include "Utils.h"

namespace pu
{
	float Scale(float value, float srcMin, float srcMax, float dstMin, float dstMax)
	{
		return (srcMin * dstMax - srcMax * dstMin + dstMin * value - dstMax * value) / (srcMin - srcMax);
	}

	cv::Mat ToDisplayable(const cv::Mat & m)
	{
		cv::Mat res{ m.rows, m.cols, CV_32FC1, cv::Scalar(0.0) };
		cv::normalize(m, res, 0.0, 1.0, cv::NORM_MINMAX);
		return res;
	}
}