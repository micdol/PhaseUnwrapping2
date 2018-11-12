#pragma once
#include "Bitflags.h"
#include <opencv2\opencv.hpp>

namespace pu
{
	namespace quality_maps
	{

		/// <summary>
		/// Computes Phase Derivative variance quality map. Variance is interpreted
		/// as deviation from average value computed in window of size KxK centered
		/// over each of the image pixels.
		/// </summary>
		/// <param name="wrapped_phase">
		/// Image with wrapped phase, 1 channel, floating point number, pixel value
		/// range [0,1].
		/// </param>
		/// <param name="k">
		/// Size of the window whole side, odd, greater equal 3. For instance k = 3 
		/// means that the window will contain 9 pixels (at most).
		/// </param>
		/// <param name="bitflags">
		/// [optional, default = null] Image with bitflags per each pixel in 
		/// wrapped phase image, same size as wrapped phase, 1 channel, pixel type 
		/// as defined by bitflag_type typedef.
		/// </param>
		/// <param name="ignore_flag">
		/// [optional, default = NoFlag] Bit-or combination of flags which  should 
		/// be ignored during computations.
		/// </param>
		/// <returns>
		/// Image with PDV, 1 channel, floating point, scaled to range [0, 1].
		/// Quality values should be read as: 0 = worst, 1 = best.
		/// </returns>
		cv::Mat PDV(const cv::Mat& wrapped_phase, int k, cv::Mat* bitflags = nullptr, Bitflag ignore_flag = Bitflag::NoFlag);
		
		/// <summary>
		/// Computes Maximum Gradient quality map. Maximum gradient is computed as
		/// maximum absolute value of both Dx and Dy gradients in window of size
		/// KxK centered over each of the image pixels. 
		/// </summary>
		/// <param name="wrapped_phase">
		/// Image with wrapped phase, 1 channel, floating point number, pixel value
		/// range [0,1].
		/// </param>
		/// <param name="k">
		/// Size of the window whole side, odd, greater equal 3. For instance k = 3 
		/// means that the window will contain 9 pixels (at most).
		/// </param>
		/// <param name="bitflags">
		/// [optional, default = null] Image with bitflags per each pixel in 
		/// wrapped phase image, same size as wrapped phase, 1 channel, pixel type 
		/// as defined by bitflag_type typedef.
		/// </param>
		/// <param name="ignore_flag">
		/// [optional, default = NoFlag] Bit-or combination of flags which  should 
		/// be ignored during computations.
		/// </param>
		/// <returns>
		/// Image with Maximum Absolute Gradient, 1 channel, floating point, scaled
		/// to range [0, 1]. Quality values should be read as: 0 = worst, 1 = best.
		/// </returns>
		cv::Mat MaxAbsGrad(const cv::Mat& wrapped_phase, int k, cv::Mat* bitflags = nullptr, Bitflag ignore_flag = Bitflag::NoFlag);

		namespace
		{
			/// <summary>
			/// Computes windowed variance within window size of KxK around each 
			/// pixel.
			/// </summary>
			/// <param name="image"></param>
			/// <param name="k"></param>
			/// <param name="bitflags"></param>
			/// <param name="ignore_flag"></param>
			/// <returns></returns>
			cv::Mat WindowedVariance(const cv::Mat& image, int k, cv::Mat* bitflags = nullptr, Bitflag ignore_flag = Bitflag::NoFlag);

			/// <summary>
			/// Computes windowed maximum absolute value within window size of KxK 
			/// around each pixel.
			/// </summary>
			/// <param name="image"></param>
			/// <param name="k"></param>
			/// <param name="bitflags"></param>
			/// <param name="ignore_flag"></param>
			/// <returns></returns>
			cv::Mat WindowedMaxAbs(const cv::Mat& image, int k, cv::Mat* bitflags = nullptr, Bitflag ignore_flag = Bitflag::NoFlag);
		}
	}
}
