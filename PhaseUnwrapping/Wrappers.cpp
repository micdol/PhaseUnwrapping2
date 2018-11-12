#include "Wrappers.h"

namespace pu
{
	float Wrap(float phase)
	{
		return std::atan2f(std::sinf(phase), std::cosf(phase));
	}

	cv::Mat Wrap(const cv::Mat & phase, bool normalize)
	{
		// Create image same size as input
		cv::Mat res{ phase.rows, phase.cols, CV_32FC1, cv::Scalar(0.0) };

		// Wrap each pixel
		res.forEach<float>([&](float & px, const int * pos) -> void {
			int row = pos[0], col = pos[1];
			px = Wrap(phase.at<float>(row, col));
		});

		// Normalize if necessary
		if(normalize)
		{
			cv::normalize(res, res, 0.0, 1.0, cv::NORM_MINMAX);
		}

		return res;
	}

}
