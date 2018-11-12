#include "QualityMaps.h"
#include "Gradients.h"

#include <mutex>

namespace pu
{
	namespace quality_maps
	{
		cv::Mat PDV(const cv::Mat & wrapped_phase, int k, cv::Mat * bitflags, Bitflag ignore_flag)
		{
			assert(!wrapped_phase.empty() &&
				   wrapped_phase.type() == CV_32FC1 &&
				   "[PDV] Invalid wrapped phase image");

			if(bitflags)
			{
				assert(!bitflags->empty() &&
					   bitflags->type() == CV_MAKETYPE(cv::DataType<bitflag_type>::type, 1) &&
					   bitflags->size() == wrapped_phase.size() &&
					   "[PDV] Invalid bitflags image");
			}
			assert(k >= 3 &&
				(k % 2) == 1 &&
				   "[MaxAbsGrad] K must be odd, greater equal 3");

			// First compute dx derivative
			cv::Mat dx = DxGradient(wrapped_phase);

			// Image for the results	
			cv::Mat pdv = WindowedVariance(dx, k, bitflags, ignore_flag);

			// Then compute dy derivative;
			cv::Mat dy = DyGradient(wrapped_phase);

			// And add dy variance to the results
			pdv += WindowedVariance(dy, k, bitflags, ignore_flag);

			// Scale image to range [0,1]
			// Since higher variance indicates worse pixels it needs to be inverted
			cv::normalize(-pdv, pdv, 0, 1, cv::NORM_MINMAX);

			return pdv;
		}

		cv::Mat MaxAbsGrad(const cv::Mat & wrapped_phase, int k, cv::Mat * bitflags, Bitflag ignore_flag)
		{
			assert(!wrapped_phase.empty() &&
				   wrapped_phase.type() == CV_32FC1 &&
				   "[MaxAbsGrad] Invalid wrapped phase image");

			if(bitflags)
			{
				assert(!bitflags->empty() &&
					   bitflags->type() == CV_MAKETYPE(cv::DataType<bitflag_type>::type, 1) &&
					   bitflags->size() == wrapped_phase.size() &&
					   "[MaxAbsGrad] Invalid bitflags image");
			}
			assert(k >= 3 &&
				(k % 2) == 1 &&
				   "[MaxAbsGrad] K must be odd, greater equal 3");

			// Compute dx derivative
			cv::Mat dx = DxGradient(wrapped_phase);

			// Check maximum gradients for dx derivative	
			cv::Mat maxgradx = WindowedMaxAbs(dx, k, bitflags, ignore_flag);

			// Compute dy derivative;
			cv::Mat dy = DyGradient(wrapped_phase);

			// Check maximum gradients for dy derivative	
			cv::Mat maxgrady = WindowedMaxAbs(dy, k, bitflags, ignore_flag);

			// Image for results - per pixel maximum of either maxgradx or maxgrady
			cv::Mat maxgrad;
			cv::max(maxgradx, maxgrady, maxgrad);

			// Scale image to range [0,1]
			// Since higher variance indicates bad pixels it needs to be inverted
			cv::normalize(-maxgrad, maxgrad, 0, 1, cv::NORM_MINMAX);

			// Since higher gradient indicates bad pixels it needs to be inverted
			return maxgrad;
		}

		namespace
		{
			cv::Mat WindowedVariance(const cv::Mat & image, int k, cv::Mat* bitflags, Bitflag ignore_flag)
			{
				// Only floating point images are supported
				assert(!image.empty() &&
					   image.type() == CV_32FC1 &&
					   "[WindowedVariance] Invalid image");

				// If user provided bitflags (double) check that it has proper size
				if(bitflags)
				{
					assert(bitflags->size() == image.size() &&
						   "[WindowedVariance] Invalid bitflags image");
				}

				// Dimensions and image rect
				int rows = image.rows, cols = image.cols;
				cv::Rect rect{ 0,0,cols, rows };

				// Half window size rounded down
				int k2 = k / 2;

				// Image for the results
				cv::Mat variance{ rows,cols,CV_32FC1,cv::Scalar(0) };

				// Compute dx derivative variance over window of size k and place it in result image
				variance.forEach<float>([&](float &var_px, const int* pos) -> void {
					// Coordinates of center pixel
					int row = pos[0], col = pos[1];

					// Intersection of image and window centered at current pixel (clipping out of bounds pixels at edges)
					// Ignored bitflags (if any) will be considered later
					auto roi = rect & cv::Rect(col - k2, row - k2, k, k);

					// "Cut" the window of size k around [row,col] from dx derivative (may vary at edges)
					cv::Mat window = image(roi);

					// Variance = avgsqr - avg*avg
					float avgsqr = 0, avg = 0;

					// n will vary depending on window placement (edges, corners) and optional pixels marked to be ignored with bitflags
					int n = 0;

					// Just to speed things up a bit make one initial check instead of in each loop 
					// With bitflag checking
					if(bitflags && ignore_flag != Bitflag::NoFlag)
					{
						window.forEach<float>([&](float& win_px, const int* pos) -> void {
							// Coordinates of window pixel (in IMAGE pixel coordinates reference)
							int winRow = row + pos[0] - k2, winCol = col + pos[1] - k2;

							// Check for ignore code in bitflags
							if(bitflags->at<bitflag_type>(winRow, winCol) & ignore_flag)
							{
								return;
							}

							avg += win_px;
							avgsqr += win_px * win_px;
							n++;
						});
					}
					// Without bitflag checking
					else
					{
						window.forEach<float>([&](float& win_px, const int* pos) -> void {
							avg += win_px;
							avgsqr += win_px * win_px;
							n++;
						});
					}

					// To avoid division by zero and also to zero empty widnows
					float m = n > 0 ? 1.0 / n : 0.0;
					avgsqr *= m;
					avg *= m;

					var_px = avgsqr - avg*avg;
				});

				return variance;
			}

			cv::Mat WindowedMaxAbs(const cv::Mat & image, int k, cv::Mat* bitflags, Bitflag ignore_flag)
			{
				// Only floating point images are supported
				assert(!image.empty() &&
					   image.type() == CV_32FC1 &&
					   "[WindowedMaxAbs] Invalid image");

				// If user provided bitflags (double) check that it has proper size
				if(bitflags)
				{
					assert(bitflags->size() == image.size() &&
						   "[WindowedMaxAbs] Invalid bitflags image");
				}

				// Dimensions and image rect
				int rows = image.rows, cols = image.cols;
				cv::Rect rect{ 0,0,cols, rows };

				// Half window size rounded down
				int k2 = k / 2;

				// Image for the results
				cv::Mat maximum{ rows,cols,CV_32FC1,cv::Scalar(0) };

				// Compute dx derivative variance over window of size k and place it in result image
				maximum.forEach<float>([&](float &max_px, const int* pos) -> void {
					int row = pos[0], col = pos[1];

					// Intersection of image and window centered at current pixel
					auto roi = rect & cv::Rect(col - k2, row - k2, k, k);

					// "Cut" the window from dx derivative
					cv::Mat window = image(roi);

					// Just to speed things up a bit make one initial check instead of in each loop 
					// With bitflag checking
					if(bitflags && ignore_flag != Bitflag::NoFlag)
					{
						std::mutex mtx;
						window.forEach<float>([&](float& win_px, const int* pos) -> void {
							// Coordinates of window pixel (in IMAGE pixel coordinates reference)
							int winRow = row + pos[0] - k2, winCol = col + pos[1] - k2;

							// Check for ignore code in bitflags
							if(bitflags->at<bitflag_type>(winRow, winCol) & ignore_flag)
							{
								return;
							}

							// Since forEach is multi threaded and updating max is not atomic lock need to be issued...
							std::lock_guard<std::mutex> lock(mtx);
							max_px = std::max(std::abs(win_px), max_px);
						});

					}
					// Without bitflag checking
					else
					{
						// Compute maximum in the given window
						double minVal, maxVal;
						cv::minMaxLoc(window, &minVal, &maxVal);
						max_px = static_cast<float>(std::max(std::abs(minVal), std::abs(maxVal)));
					}

				});

				return maximum;
			}
		}
	}
}