#include "Gradients.h"

namespace pu
{
	float Gradient(float current, float other)
	{
		float r = current - other;

		// To make gradient smooth, so that wraps which may correspond to 
		// large jumps (> PI a.k.a 0.5 since scaling) are smoothed out		
		if(r > 0.5f) r -= 1.0f;
		else if(r < -0.5f) r += 1.0f;

		return r;
	}

	cv::Mat DxGradient(const cv::Mat & wrapped_phase)
	{
		assert(!wrapped_phase.empty() &&
			   wrapped_phase.type() == CV_32FC1 &&
			   "[DxGradient] Invalid wrapped phase image");

		int rows = wrapped_phase.rows, cols = wrapped_phase.cols;

		// Result image
		cv::Mat dx{ rows, cols, CV_32FC1, cv::Scalar(0.0) };

		wrapped_phase.forEach<float>([&](const float& px, const int* pos) -> void {
			int row = pos[0], col = pos[1];

			// For pixels not last in row take diff: next - curr 
			if(col < cols - 1)
			{
				float current = wrapped_phase.at<float>(row, col + 1);
				dx.at<float>(row, col) = Gradient(current, px);
			}
			// For the last pixels in row take diff: prev - curr
			else
			{
				float current = wrapped_phase.at<float>(row, col - 1);
				dx.at<float>(row, col) = Gradient(px, current);
			}
		});

		return dx;
	}

	cv::Mat DyGradient(const cv::Mat & wrapped_phase)
	{
		assert(!wrapped_phase.empty() &&
			   wrapped_phase.type() == CV_32FC1 &&
			   "[DyGradient] Invalid wrapped phase image");

		int rows = wrapped_phase.rows, cols = wrapped_phase.cols;

		// Result image
		cv::Mat dy{ rows, cols, CV_32FC1, cv::Scalar(0.0) };

		wrapped_phase.forEach<float>([&](const float& px, const int* pos) -> void {
			int row = pos[0], col = pos[1];

			// For pixels not last in col take diff: next- curr 
			if(row < rows - 1)
			{
				float current = wrapped_phase.at<float>(row + 1, col);
				dy.at<float>(row, col) = Gradient(current, px);
			}
			// For the last pixels in col take diff: prev - curr
			else
			{
				float current = wrapped_phase.at<float>(row - 1, col);
				dy.at<float>(row, col) = Gradient(px, current);
			}
		});

		return dy;
	}
}