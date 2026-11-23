// Copyright (c) 2016-2021 Memgraph Ltd. [https://memgraph.com]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//! Macro definitions and top level data structures.
//!
//! [define_procedure] accepts a function taking [&Memgraph], [init_module] accepts a function
//! taking [&MemgraphModule], and [close_module] accepts a zero-argument function, all returning
//! [crate::result::Result] so callers can propagate execution errors with `?`.

use std::any::Any;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};

use crate::memgraph::*;
#[double]
use crate::mgp::ffi;
use crate::mgp::mgp_result;
use crate::result::{Error, Result};
use crate::value::*;
use mockall_double::double;

const FALLBACK_ERROR_MSG: &[u8] = b"Procedure execution failed.\0";

/// Defines a new procedure callable by Memgraph engine.
///
/// Wraps a call to the Rust function provided as the second argument with `catch_unwind` and
/// panic-free error reporting.
#[macro_export]
macro_rules! define_procedure {
    ($c_name:ident, $rs_func:expr) => {
        #[no_mangle]
        pub extern "C" fn $c_name(
            args: *mut $crate::mgp::mgp_list,
            graph: *mut $crate::mgp::mgp_graph,
            result: *mut $crate::mgp::mgp_result,
            memory: *mut $crate::mgp::mgp_memory,
        ) {
            let procedure_result =
                ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                    let memgraph =
                        unsafe { $crate::memgraph::Memgraph::new(args, graph, result, memory) };
                    $rs_func(&memgraph).map_err(|err| err.to_string())
                }));

            match procedure_result {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    $crate::rsmgp::report_raw_result_error(result, &msg);
                }
                Err(panic_payload) => {
                    let msg = $crate::rsmgp::panic_payload_to_str(panic_payload.as_ref());
                    $crate::rsmgp::report_raw_result_error(result, msg);
                }
            }
        }
    };
}

/// Initializes Memgraph query module.
#[macro_export]
macro_rules! init_module {
    ($init_func:expr) => {
        #[no_mangle]
        pub extern "C" fn mgp_init_module(
            module: *mut $crate::mgp::mgp_module,
            memory: *mut $crate::mgp::mgp_memory,
        ) -> ::std::os::raw::c_int {
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let memgraph = unsafe { $crate::memgraph::MemgraphModule::new(module, memory) };
                $init_func(&memgraph)
            }));
            match result {
                Ok(Ok(())) => 0,
                Ok(Err(_)) | Err(_) => 1,
            }
        }
    };
}

/// Closes Memgraph query module.
#[macro_export]
macro_rules! close_module {
    ($close_func:expr) => {
        #[no_mangle]
        pub extern "C" fn mgp_shutdown_module() -> ::std::os::raw::c_int {
            let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                $close_func()
            }));
            match result {
                Ok(Ok(())) => 0,
                Ok(Err(_)) | Err(_) => 1,
            }
        }
    };
}

/// Used to pass expected types (arguments, optional arguments, result field types) to the Memgraph
/// engine.
pub enum Type {
    Any,
    Bool,
    Number,
    Int,
    Double,
    String,
    Map,
    Vertex,
    Edge,
    Path,
    Nullable,
    List,
    Date,
    LocalTime,
    LocalDateTime,
    Duration,
}

/// Used to pass expected type during procedure registration for required parameters and result
/// values.
pub struct NamedType<'a> {
    pub name: &'a CStr,
    pub types: &'a [Type],
    pub deprecated: bool,
}

/// Used to pass expected type during procedure registration for optional parameters.
pub struct OptionalNamedType<'a> {
    pub name: &'a CStr,
    pub types: &'a [Type],
    pub default: &'a MgpValue,
}

/// Defines a regular type.
#[macro_export]
macro_rules! define_type {
    ($name:literal, $($types:expr),+) => {
        NamedType {
            name: &c_str!($name),
            types: &[$($types),+],
            deprecated: false,
        }
    };
}

/// Defines a deprecated type.
#[macro_export]
macro_rules! define_deprecated_type {
    ($name:literal, $($types:expr),+) => {
        NamedType {
            name: &c_str!($name),
            types: &[$($types),+],
            deprecated: true,
        }
    };
}

/// Defines an optional type.
#[macro_export]
macro_rules! define_optional_type {
    ($name:literal, $default:expr, $($types:expr),+) => {
        OptionalNamedType {
            name: &c_str!($name),
            types: &[$($types),+],
            default: $default,
        }
    };
}

#[doc(hidden)]
pub fn panic_payload_to_str(payload: &(dyn Any + Send)) -> &str {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.as_str()
    } else {
        "Procedure panicked with an unknown payload."
    }
}

unsafe fn set_raw_error_cstr(result: *mut mgp_result, msg: *const c_char) -> Result<()> {
    invoke_void_mgp_func_with_res!(
        Error::UnableToSetErrorMessage,
        ffi::mgp_result_set_error_msg,
        result,
        msg
    )
}

#[doc(hidden)]
pub fn report_raw_result_error(result: *mut mgp_result, msg: &str) {
    #[cfg(not(test))]
    if result.is_null() {
        return;
    }
    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
        let bytes = msg.as_bytes();
        let c_candidate = if !bytes.contains(&0) {
            CString::new(bytes).ok()
        } else {
            let sanitized: Vec<u8> = bytes
                .iter()
                .map(|&b| if b == 0 { b' ' } else { b })
                .collect();
            CString::new(sanitized).ok()
        };
        if let Some(c_msg) = c_candidate {
            if unsafe { set_raw_error_cstr(result, c_msg.as_ptr()) }.is_ok() {
                return;
            }
        }
        unsafe {
            let _ = ffi::mgp_result_set_error_msg(
                result,
                FALLBACK_ERROR_MSG.as_ptr() as *const c_char,
            );
        }
    }));
}

/// Fallibly sets the error message that will be returned to the caller without panicking.
pub fn try_set_memgraph_error_msg(msg: &CStr, memgraph: &Memgraph) -> Result<()> {
    #[cfg(not(test))]
    if memgraph.result_ptr().is_null() {
        return Err(Error::UnableToSetErrorMessage);
    }
    unsafe { set_raw_error_cstr(memgraph.result_ptr(), msg.as_ptr()) }
}

/// Sets the error message that will be returned to the caller, sanitizing interior NUL bytes and
/// never panicking across FFI.
pub fn set_memgraph_error_str(msg: &str, memgraph: &Memgraph) {
    report_raw_result_error(memgraph.result_ptr(), msg);
}

/// Sets error that will be returned to the caller without panicking on allocation failure.
pub fn set_memgraph_error_msg(msg: &CStr, memgraph: &Memgraph) {
    if try_set_memgraph_error_msg(msg, memgraph).is_err() {
        unsafe {
            let _ = ffi::mgp_result_set_error_msg(
                memgraph.result_ptr(),
                FALLBACK_ERROR_MSG.as_ptr() as *const c_char,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use c_str_macro::c_str;
    use serial_test::serial;

    use super::*;
    use crate::mgp::mock_ffi::*;
    use crate::{mock_mgp_once, with_dummy};

    #[test]
    #[serial]
    fn test_set_error_msg() {
        mock_mgp_once!(mgp_result_set_error_msg_context, |_, _| {
            crate::mgp::mgp_error::MGP_ERROR_NO_ERROR
        });

        with_dummy!(|memgraph: &Memgraph| {
            set_memgraph_error_msg(c_str!("test_error"), &memgraph);
        });
    }
}
