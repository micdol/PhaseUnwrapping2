#pragma once

namespace pu
{
	/// <summary>
	/// Typedef specifing type of the bitflag, 
	/// used in several other files mostly for image type assertions
	/// </summary>
	typedef unsigned short bitflag_type;

	/// <summary>
	/// All available bitflags, in general all should be exclusive
	/// however since different algorithms will use different flags
	/// (at most 8) this assertion need to hold only in the ones used
	/// </summary>
	enum Bitflag : bitflag_type
	{
		/// <summary>
		/// Indicates that no flags are used, for algorithms readability reasons
		/// </summary>
		NoFlag = 0x0000,

		/// <summary>
		/// Indicates image border
		/// </summary>
		Border = 0x0001
	};
}
