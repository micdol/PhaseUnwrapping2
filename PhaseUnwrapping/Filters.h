#pragma once
#include <opencv2\opencv.hpp>

namespace pu
{
	namespace filters
	{
		/// <summary>
		/// Computes "mean" phase filter. It recomputes complex Re and Im
		/// coeficients from phase, calculates their mean (in each window)
		/// and finally recomputes "mean" value of phase.
		/// </summary>
		/// <param name="wrapped">
		/// Image with the wrapped phase, single channel, floating point,
		/// values should be in range [0, 1]
		/// </param>
		/// <param name="k">
		/// Size of window, must be odd, greater or equal 3
		/// </param>
		/// <returns>
		/// Filtered image, single channel, floating point noramlized to
		/// range [0, 1]
		/// </returns>
		cv::Mat MeanPhaseFilter(const cv::Mat& wrapped, int k);

		/// <summary>
		/// Computes "median" phase filter. It recomputes complex Re and Im
		/// coeficients from phase, calculates their median (in each window)
		/// and finally recomputes "mean" value of phase.
		/// </summary>
		/// <param name="wrapped">
		/// Image with the wrapped phase, single channel, floating point,
		/// values should be in range [0, 1]
		/// </param>
		/// <param name="k">
		/// Size of window, must be odd, greater or equal 3
		/// </param>
		/// <returns>
		/// Filtered image, single channel, floating point noramlized to
		/// range [0, 1]
		/// </returns>
		cv::Mat MedianPhaseFilter(const cv::Mat& wrapped, int k);
	}
}