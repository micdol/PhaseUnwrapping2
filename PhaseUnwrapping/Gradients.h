#pragma once
#include <opencv2\opencv.hpp>

namespace pu
{
	/// <summary>
	/// Computes gradient of wrapped phase between current and other. 
	/// </summary>
	/// <param name="current">
	/// "Reference" phase value (to subtract from). Should lay in range [0, 1].
	/// </param>
	/// <param name="other">
	/// Phase value being subtracted. Should lay in range [0, 1].
	/// </param>
	/// <returns>
	/// Phase gradient, in range [0, 1].
	/// </returns>
	float Gradient(float current, float other);

	/// <summary>
	/// Computes wrapped phase gradient in 0X direction (each row).
	/// </summary>
	/// <param name="wrapped_phase">
	/// Wrapped phase image, not empty, single channel, floating point.
	/// Phase values should lay in range [0, 1].
	/// </param>
	/// <returns>
	/// Image with computed gradient, single channel, floating point,
	/// same size as input image. Values lay in range [0, 1].
	/// </returns>
	cv::Mat DxGradient(const cv::Mat & wrapped_phase);

	/// <summary>
	/// Computes wrapped phase gradient in OY direction (each column).
	/// </summary>
	/// <param name="wrapped_phase">
	/// Wrapped phase image, not empty, single channel, floating point.
	/// Phase values should lay in range [0, 1].
	/// </param>
	/// <returns>
	/// Image with computed gradient, single channel, floating point,
	/// same size as input image. Values lay in range [0, 1].
	/// </returns>
	cv::Mat DyGradient(const cv::Mat & wrapped_phase);
}