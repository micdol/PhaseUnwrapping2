#include "Filters.h"

namespace pu
{
	namespace filters
	{
		cv::Mat MeanPhaseFilter(const cv::Mat & wrapped, int k)
		{
			assert(!wrapped.empty() &&
				   wrapped.type() == CV_32FC1 &&
				   "[MeanPhaseFilter] Invalid wrapped phase image");

			assert(k >= 3 &&
				(k % 2) == 1 &&
				   "[MeanPhaseFilter] K must be odd, greater equal 3");

			// Dimensions and image rect
			int rows = wrapped.rows, cols = wrapped.cols;
			cv::Rect rect{ 0,0,cols, rows };

			// Half window size rounded down
			int k2 = k / 2;

			// Image for the result
			cv::Mat filtered{ rows, cols, CV_32FC1, cv::Scalar(0) };

			wrapped.forEach<float>([&](const float& px, const int * pos) -> void {
				int row = pos[0], col = pos[1];

				// "Cut out" the region of interest
				cv::Rect roi = rect & cv::Rect(col - k2, row - k2, k, k);
				cv::Mat window = wrapped(roi);

				// Recompute Re and Im from Phase in the window
				cv::Mat cplx{ window.rows, window.cols, CV_32FC2, cv::Scalar(0) };
				window.forEach<float>([&](const float& win_px, const int * win_pos) -> void {
					int win_row = win_pos[0], win_col = win_pos[1];
					// Phase is in range [0,1]
					float phase = win_px * CV_PI * 2.0f;
					float re = std::cosf(phase);
					float im = std::sinf(phase);
					cplx.at<cv::Vec2f>(win_row, win_col) = cv::Vec2f(re, im);
				});

				// Compute mean 
				cv::Scalar mean = cv::mean(cplx);
				float re = mean[0];
				float im = mean[1];

				filtered.at<float>(row, col) = std::atan2f(im, re);
			});
			
			// Scale result to [0,1] range
			cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);

			return filtered;
		}

		cv::Mat MedianPhaseFilter(const cv::Mat & wrapped, int k)
		{
			assert(!wrapped.empty() &&
				   wrapped.type() == CV_32FC1 &&
				   "[MeanPhaseFilter] Invalid wrapped phase image");

			assert(k >= 3 &&
				(k % 2) == 1 &&
				   "[MeanPhaseFilter] K must be odd, greater equal 3");

			// Dimensions and image rect
			int rows = wrapped.rows, cols = wrapped.cols;
			cv::Rect rect{ 0,0,cols, rows };

			// Half window size rounded down
			int k2 = k / 2;

			// Image for the result
			cv::Mat filtered{ rows, cols, CV_32FC1, cv::Scalar(0) };

			wrapped.forEach<float>([&](const float& px, const int * pos) -> void {
				int row = pos[0], col = pos[1];

				// "Cut out" the region of interest
				cv::Rect roi = rect & cv::Rect(col - k2, row - k2, k, k);
				cv::Mat window = wrapped(roi);

				// Recompute Re and Im from Phase in the window
				cv::Mat re_part{ window.rows, window.cols, CV_32FC1, cv::Scalar(0) };
				cv::Mat im_part{ window.rows, window.cols, CV_32FC1, cv::Scalar(0) };
				window.forEach<float>([&](const float& win_px, const int * win_pos) -> void {
					int win_row = win_pos[0], win_col = win_pos[1];
					// Phase is in range [0,1]
					float phase = win_px * CV_PI * 2.0f;
					float re = std::cosf(phase);
					float im = std::sinf(phase);
					re_part.at<float>(win_row, win_col) = re;
					im_part.at<float>(win_row, win_col) = im;
				});

				int n2 = window.total() / 2;
				
				// Partialy sort Re and Im so that middle element(s) (needed for median) are on their place
				auto re_begin = re_part.begin<float>();
				auto re_mid = re_part.begin<float>() + n2 + 1;
				auto re_end = re_part.end<float>();
				std::nth_element(re_begin, re_mid, re_end);

				auto im_begin = im_part.begin<float>();
				auto im_mid = im_part.begin<float>() + n2 + 1;
				auto im_end = im_part.end<float>();
				std::nth_element(im_begin, im_mid, im_end);

				// Compute Re and Im median
				// Odd number of pixels - take only middle element 
				float re, im;
				if(window.total() % 2)
				{
					re = *(re_part.begin<float>() + n2);
					im = *(im_part.begin<float>() + n2);
				}
				// Even number of pixels - take average of two middle elements
				else
				{
					re = (*(re_part.begin<float>() + n2) + *(re_part.begin<float>() + n2 + 1)) / 2.0f;
					im = (*(im_part.begin<float>() + n2) + *(im_part.begin<float>() + n2 + 1)) / 2.0f;
				}

				// Compute median phase
				filtered.at<float>(row, col) = std::atan2f(im, re);
			});

			// Scale result to [0,1] range
			cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);

			return filtered;
		}
	}
}
