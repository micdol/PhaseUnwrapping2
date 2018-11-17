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

			cv::Mat cplx{ rows,cols, CV_32FC2, cv::Scalar(0) };

			// Re-map phase into complex values
			wrapped.forEach<float>([&](const float & px, const int * pos) -> void {
				int row = pos[0], col = pos[1];

				// Wrapped phase is in range [0,1]
				float phase = px * CV_PI * 2.0;
				float re = std::cosf(phase);
				float im = std::sinf(phase);
				cplx.at<cv::Vec2f>(row, col) = cv::Vec2f(re, im);
			});

			cv::Mat filtered{ rows, cols, CV_32FC1, cv::Scalar(0) };

			// Compute windowed mean around each pixel and store result in filtered
			filtered.forEach<float>([&](float & px, const int * pos) -> void {
				int row = pos[0], col = pos[1];

				// Intersection of image and window centered at current pixel
				auto roi = rect & cv::Rect(col - k2, row - k2, k, k);

				cv::Mat window = cplx(roi);

				// Compute mean per each channel
				cv::Scalar result = cv::mean(window);
				float re = result[0];
				float im = result[1];

				// Retrieve "mean" phase
				px = std::atan2f(im, re);
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

			cv::Mat cplx{ rows,cols, CV_32FC2, cv::Scalar(0) };

			// Re-map phase into complex values
			wrapped.forEach<float>([&](const float & px, const int * pos) -> void {
				int row = pos[0], col = pos[1];

				// Wrapped phase is in range [0,1]
				float phase = px * CV_PI * 2.0;
				float re = std::cosf(phase);
				float im = std::sinf(phase);
				cplx.at<cv::Vec2f>(row, col) = cv::Vec2f(re, im);
			});

			cv::Mat filtered{ rows, cols, CV_32FC1, cv::Scalar(0) };

			// Compute windowed mean around each pixel and store result in filtered
			filtered.forEach<float>([&](float & px, const int * pos) -> void {
				int row = pos[0], col = pos[1];

				// Intersection of image and window centered at current pixel
				auto roi = rect & cv::Rect(col - k2, row - k2, k, k);

				cv::Mat window = cplx(roi);

				// Compute median per each channel (needs cloning due to sorting)
				std::vector<cv::Mat> reim;
				cv::split(window.clone(), reim);
				int n = window.total();
				float re, im;
				if(n % 2)
				{
					std::nth_element(reim[0].begin<float>(), reim[0].begin<float>() + n / 2, reim[0].end<float>());
					std::nth_element(reim[1].begin<float>(), reim[1].begin<float>() + n / 2, reim[1].end<float>());
					re = *(reim[0].begin<float>() + n / 2);
					im = *(reim[1].begin<float>() + n / 2);
				}
				else
				{
					std::nth_element(reim[0].begin<float>(), reim[0].begin<float>() + n / 2 + 1, reim[0].end<float>());
					std::nth_element(reim[1].begin<float>(), reim[1].begin<float>() + n / 2 + 1, reim[1].end<float>());
					re = (*(reim[0].begin<float>() + n / 2) + *(reim[0].begin<float>() + n / 2)) / 2;
					im = (*(reim[1].begin<float>() + n / 2) + *(reim[1].begin<float>() + n / 2)) / 2;
				}

				// Retrieve "median" phase
				px = std::atan2f(im, re);
			});

			// Scale result to [0,1] range
			cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);

			return filtered;
		}
	}
}
