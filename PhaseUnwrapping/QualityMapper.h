#pragma once
#include "Bitflags.h"
#include <opencv2\opencv.hpp>

namespace pu
{
	/// <summary>
	/// Class encapsulating logic and computation of quality maps.
	/// </summary>
	class QualityMapper
	{
	private:
		/// <summary>
		/// Wrapped phase image, data is assured not to be modified 
		/// during any computations however have any other external 
		/// code modify the referenced image results will change as
		/// input data will change (it's a shallow copy, Mat header
		/// only, data is shared)
		/// </summary>
		cv::Mat m_wrapped_phase;

		/// <summary>
		/// [Optional] Bitflag image, contains bitflags corresponding
		/// to each pixel in wrappe phase image if provide
		/// </summary>
		cv::Mat* m_bitflags;

		/// <summary>
		/// Code of flag which is marking pixels which should not be considered
		/// during computations (they will be skipped)
		/// </summary>
		bitflag_type m_ignore_flag;

	public:

		/// <summary>
		/// Initializes new QualityMapper insance
		/// </summary>
		/// <param name="wrapped_phase">
		/// cv::Mat with wrapped phase, image should contain 1 channel,
		/// pixel type should be single floating point number in range [0,1]
		/// </param>
		/// <param name="bitflags">
		/// [optional, default = null] cv::Mat with bitflags per each pixel 
		/// in wrapped phase image, should be the same size as wrapped phase,
		/// contain 1 channel and pixel type should be as defined by 
		/// bitflag_type typedef.
		/// </param>
		/// <param name="ignore_flag">
		/// [optional, default = NoFlag] Bit-or combination of flags which 
		/// should be ignored during computations
		/// </param>
		QualityMapper(const cv::Mat& wrapped_phase, 
					  cv::Mat* bitflags = nullptr, 
					  const bitflag_type& ignore_flag = Bitflag::NoFlag);
		
		/// <summary>
		/// Destructor, since no instance-scope allocations are done
		/// default implementation will do
		/// </summary>
		~QualityMapper() = default;

		/// <summary>
		/// Computes Phase Derivative variance quality map. Variance is interpreted
		/// as deviation from average value computed in window of size KxK centered
		/// over each of the image pixels. As input image provided wrapped phase is
		/// taken.
		/// </summary>
		/// <param name="k">
		/// Size of the window, odd, greater equal 3.
		/// </param>
		/// <returns>
		/// Image with PDV, scaled to range [0, 1], 0 = worst, 1 = best.
		/// </returns>
		cv::Mat PDV(int k) const;

		/// <summary>
		/// Computes Maximum Gradient quality map. Maximum gradient is computed as
		/// maximum absolute value of both Dx and Dy gradients in window of size
		/// KxK centered over each of the image pixels. 
		/// </summary>
		/// <param name="k">
		/// Size of the window, odd, greater equal 3.
		/// </param>
		/// <returns>
		/// Image with Maximum Gradient, scaled to range [0, 1], 0 = worst, 1 = best.
		/// </returns>
		cv::Mat MaxGrad(int k) const;

	private:
		cv::Mat WindowedVariance(const cv::Mat& image, int k) const;
		cv::Mat WindowedMaxAbs(const cv::Mat& image, int k) const;
	};
}
