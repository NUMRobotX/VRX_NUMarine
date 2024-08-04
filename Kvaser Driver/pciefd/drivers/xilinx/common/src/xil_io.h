/******************************************************************************
* Copyright (c) 2014 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/*****************************************************************************/
/**
*
* @file xil_io.h
*
* @addtogroup common_io_interfacing_apis Register IO interfacing APIs
*
* The xil_io.h file contains the interface for the general I/O component, which
* encapsulates the Input/Output functions for the processors that do not
* require any special I/O handling.
*
* @{
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who      Date     Changes
* ----- -------- -------- -----------------------------------------------
* 5.00 	pkp  	 05/29/14 First release
* 6.00  mus      08/19/16 Remove checking of __LITTLE_ENDIAN__ flag for
*                         ARM processors
* 7.20  har      01/03/20 Added Xil_SecureOut32 for avoiding blindwrite for
*                         CR-1049218
* </pre>
******************************************************************************/

#ifndef XIL_IO_H           /* prevent circular inclusions */
#define XIL_IO_H           /* by using protection macros */

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/

#include "xil_types.h"
#include "xstatus.h"

#if defined (__MICROBLAZE__)
#include "mb_interface.h"
#else
//#include "xpseudo_asm.h"
#endif

/************************** Function Prototypes ******************************/
u16 Xil_EndianSwap16(u16 Data);
u32 Xil_EndianSwap32(u32 Data);
#ifdef ENABLE_SAFETY
extern u32 XStl_RegUpdate(u32 RegAddr, u32 RegVal);
#endif

/***************** Macros (Inline Functions) Definitions *********************/
#if defined __GNUC__
#if defined (__MICROBLAZE__)
#  define INST_SYNC		mbar(0)
#  define DATA_SYNC		mbar(1)
# else
#  define SYNCHRONIZE_IO	dmb()
#  define INST_SYNC		isb()
#  define DATA_SYNC		dsb()
# endif
#else
# define SYNCHRONIZE_IO
# define INST_SYNC
# define DATA_SYNC
# define INST_SYNC
# define DATA_SYNC
#endif

#if defined (__GNUC__) || defined (__ICCARM__) || defined (__MICROBLAZE__)
#define INLINE inline
#else
#define INLINE __inline
#endif

/*****************************************************************************/
/**
*
* @brief    Performs an input operation for a memory location by reading
*           from the specified address and returning the 8 bit Value read from
*            that address.
*
* @param	Addr: contains the address to perform the input operation
*
* @return	The 8 bit Value read from the specified input address.

*
******************************************************************************/
static INLINE u8 Xil_In8(UINTPTR Addr)
{
	return *(volatile u8 *) Addr;
}

/*****************************************************************************/
/**
*
* @brief    Performs an input operation for a memory location by reading from
*           the specified address and returning the 16 bit Value read from that
*           address.
*
* @param	Addr: contains the address to perform the input operation
*
* @return	The 16 bit Value read from the specified input address.
*
******************************************************************************/
static INLINE u16 Xil_In16(UINTPTR Addr)
{
	return *(volatile u16 *) Addr;
}

/*****************************************************************************/
/**
*
* @brief    Performs an input operation for a memory location by
*           reading from the specified address and returning the 32 bit Value
*           read  from that address.
*
* @param	Addr: contains the address to perform the input operation
*
* @return	The 32 bit Value read from the specified input address.
*
******************************************************************************/
static INLINE u32 Xil_In32(UINTPTR Addr)
{
	return *(volatile u32 *) Addr;
}

/*****************************************************************************/
/**
*
* @brief     Performs an input operation for a memory location by reading the
*            64 bit Value read  from that address.
*
*
* @param	Addr: contains the address to perform the input operation
*
* @return	The 64 bit Value read from the specified input address.
*
******************************************************************************/
static INLINE u64 Xil_In64(UINTPTR Addr)
{
	return *(volatile u64 *) Addr;
}

/*****************************************************************************/
/**
*
* @brief    Performs an output operation for an memory location by
*           writing the 8 bit Value to the the specified address.
*
* @param	Addr: contains the address to perform the output operation
* @param	Value: contains the 8 bit Value to be written at the specified
*           address.
*
* @return	None.
*
******************************************************************************/
static INLINE void Xil_Out8(UINTPTR Addr, u8 Value)
{
	volatile u8 *LocalAddr = (volatile u8 *)Addr;
	*LocalAddr = Value;
}

/*****************************************************************************/
/**
*
* @brief    Performs an output operation for a memory location by writing the
*            16 bit Value to the the specified address.
*
* @param	Addr contains the address to perform the output operation
* @param	Value contains the Value to be written at the specified address.
*
* @return	None.
*
******************************************************************************/
static INLINE void Xil_Out16(UINTPTR Addr, u16 Value)
{
	volatile u16 *LocalAddr = (volatile u16 *)Addr;
	*LocalAddr = Value;
}

/*****************************************************************************/
/**
*
* @brief    Performs an output operation for a memory location by writing the
*           32 bit Value to the the specified address.
*
* @param	Addr contains the address to perform the output operation
* @param	Value contains the 32 bit Value to be written at the specified
*           address.
*
* @return	None.
*
******************************************************************************/
static INLINE void Xil_Out32(UINTPTR Addr, u32 Value)
{
#ifndef ENABLE_SAFETY
	volatile u32 *LocalAddr = (volatile u32 *)Addr;
	*LocalAddr = Value;
#else
	XStl_RegUpdate(Addr, Value);
#endif
}

/*****************************************************************************/
/**
*
* @brief    Performs an output operation for a memory location by writing the
*           64 bit Value to the the specified address.
*
* @param	Addr contains the address to perform the output operation
* @param	Value contains 64 bit Value to be written at the specified address.
*
* @return	None.
*
******************************************************************************/
static INLINE void Xil_Out64(UINTPTR Addr, u64 Value)
{
	volatile u64 *LocalAddr = (volatile u64 *)Addr;
	*LocalAddr = Value;
}

/*****************************************************************************/
/**
 *
 * @brief	Performs an output operation for a memory location by writing the
 *       	32 bit Value to the the specified address and then reading it
 *       	back to verify the value written in the register.
 *
 * @param	Addr contains the address to perform the output operation
 * @param	Value contains 32 bit Value to be written at the specified address
 *
 * @return	Returns Status
 *        	- XST_SUCCESS on success
 *        	- XST_FAILURE on failure
 *
 *****************************************************************************/
static INLINE u32 Xil_SecureOut32(UINTPTR Addr, u32 Value)
{
	u32 ReadReg;
	u32 ReadRegTemp;
	u32 Status = (u32)XST_FAILURE;

	Xil_Out32(Addr, Value);

	ReadReg = Xil_In32(Addr);
	ReadRegTemp = Xil_In32(Addr);

	if( (ReadReg == Value) && (ReadRegTemp == Value) ) {
		Status = (u32)XST_SUCCESS;
	}

	return Status;
}

#if defined (__MICROBLAZE__)
#ifdef __LITTLE_ENDIAN__
# define Xil_In16LE	Xil_In16
# define Xil_In32LE	Xil_In32
# define Xil_Out16LE	Xil_Out16
# define Xil_Out32LE	Xil_Out32
# define Xil_Htons	Xil_EndianSwap16
# define Xil_Htonl	Xil_EndianSwap32
# define Xil_Ntohs	Xil_EndianSwap16
# define Xil_Ntohl	Xil_EndianSwap32
# else
# define Xil_In16BE	Xil_In16
# define Xil_In32BE	Xil_In32
# define Xil_Out16BE	Xil_Out16
# define Xil_Out32BE	Xil_Out32
# define Xil_Htons(Data) (Data)
# define Xil_Htonl(Data) (Data)
# define Xil_Ntohs(Data) (Data)
# define Xil_Ntohl(Data) (Data)
#endif
#else
# define Xil_In16LE	Xil_In16
# define Xil_In32LE	Xil_In32
# define Xil_Out16LE	Xil_Out16
# define Xil_Out32LE	Xil_Out32
# define Xil_Htons	Xil_EndianSwap16
# define Xil_Htonl	Xil_EndianSwap32
# define Xil_Ntohs	Xil_EndianSwap16
# define Xil_Ntohl	Xil_EndianSwap32
#endif

#if defined (__MICROBLAZE__)
#ifdef __LITTLE_ENDIAN__
static INLINE u16 Xil_In16BE(UINTPTR Addr)
#else
static INLINE u16 Xil_In16LE(UINTPTR Addr)
#endif
#else
static INLINE u16 Xil_In16BE(UINTPTR Addr)
#endif
{
	u16 value = Xil_In16(Addr);
	return Xil_EndianSwap16(value);
}

#if defined (__MICROBLAZE__)
#ifdef __LITTLE_ENDIAN__
static INLINE u32 Xil_In32BE(UINTPTR Addr)
#else
static INLINE u32 Xil_In32LE(UINTPTR Addr)
#endif
#else
static INLINE u32 Xil_In32BE(UINTPTR Addr)
#endif
{
	u32 value = Xil_In32(Addr);
	return Xil_EndianSwap32(value);
}

#if defined (__MICROBLAZE__)
#ifdef __LITTLE_ENDIAN__
static INLINE void Xil_Out16BE(UINTPTR Addr, u16 Value)
#else
static INLINE void Xil_Out16LE(UINTPTR Addr, u16 Value)
#endif
#else
static INLINE void Xil_Out16BE(UINTPTR Addr, u16 Value)
#endif
{
	Value = Xil_EndianSwap16(Value);
	Xil_Out16(Addr, Value);
}

#if defined (__MICROBLAZE__)
#ifdef __LITTLE_ENDIAN__
static INLINE void Xil_Out32BE(UINTPTR Addr, u32 Value)
#else
static INLINE void Xil_Out32LE(UINTPTR Addr, u32 Value)
#endif
#else
static INLINE void Xil_Out32BE(UINTPTR Addr, u32 Value)
#endif
{
	Value = Xil_EndianSwap32(Value);
	Xil_Out32(Addr, Value);
}

#ifdef __cplusplus
}
#endif

#endif /* end of protection macro */
/**
* @} End of "addtogroup common_io_interfacing_apis".
*/