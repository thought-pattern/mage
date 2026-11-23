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
//! All related to the value (container for any data type).

use std::convert::From;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::edge::*;
use crate::list::*;
use crate::map::*;
use crate::memgraph::*;
use crate::mgp::*;
use crate::path::*;
use crate::result::*;
use crate::temporal;
use crate::vertex::*;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
// Required here, if not present tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

/// Creates a copy of the provided string.
///
/// # Safety
///
/// The caller has provided a pointer that points to a valid C string. More here [CStr::from_ptr].
pub(crate) unsafe fn create_cstring(c_char_ptr: *const c_char) -> Result<CString> {
    match CString::new(CStr::from_ptr(c_char_ptr).to_bytes()) {
        Ok(v) => Ok(v),
        Err(_) => Err(Error::UnableToCreateCString),
    }
}

/// Useful to own `mgp_value` coming from / going into Memgraph as a result.
///
/// Underlying pointer object is going to be automatically deleted.
pub struct MgpValue {
    ptr: *mut mgp_value,
    memgraph: Memgraph,
}

impl Drop for MgpValue {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                ffi::mgp_value_destroy(self.ptr);
            }
        }
    }
}

impl MgpValue {
    pub(crate) fn new(ptr: *mut mgp_value, memgraph: &Memgraph) -> MgpValue {
        #[cfg(not(test))]
        assert!(
            !ptr.is_null(),
            "Unable to create Memgraph value because the given pointer is null."
        );

        MgpValue {
            ptr,
            memgraph: memgraph.clone(),
        }
    }

    pub(crate) fn mgp_ptr(&self) -> *mut mgp_value {
        self.ptr
    }

    pub fn to_value(&self) -> Result<Value> {
        unsafe { mgp_raw_value_to_value(self.mgp_ptr(), &self.memgraph) }
    }

    pub fn make_null<M: MemgraphMemory + ?Sized>(memgraph: &M) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mgp_ptr = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeNullValue,
                ffi::mgp_value_make_null,
                mg.memory_ptr()
            )?;

            Ok(MgpValue::new(mgp_ptr, &mg))
        }
    }

    pub fn is_null(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_null, self.ptr).unwrap() != 0
        }
    }

    pub fn make_bool<M: MemgraphMemory + ?Sized>(value: bool, memgraph: &M) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mgp_ptr = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeBoolValue,
                ffi::mgp_value_make_bool,
                if !value { 0 } else { 1 },
                mg.memory_ptr()
            )?;
            Ok(MgpValue::new(mgp_ptr, &mg))
        }
    }

    pub fn is_bool(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_bool, self.ptr).unwrap() != 0
        }
    }

    pub fn make_int<M: MemgraphMemory + ?Sized>(value: i64, memgraph: &M) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mgp_ptr = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeIntegerValue,
                ffi::mgp_value_make_int,
                value,
                mg.memory_ptr()
            )?;
            Ok(MgpValue::new(mgp_ptr, &mg))
        }
    }

    pub fn is_int(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_int, self.ptr).unwrap() != 0
        }
    }

    pub fn make_double<M: MemgraphMemory + ?Sized>(value: f64, memgraph: &M) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mgp_ptr = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeDoubleValue,
                ffi::mgp_value_make_double,
                value,
                mg.memory_ptr()
            )?;
            Ok(MgpValue::new(mgp_ptr, &mg))
        }
    }

    pub fn is_double(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_double, self.ptr).unwrap()
                != 0
        }
    }

    pub fn make_string<M: MemgraphMemory + ?Sized>(
        value: &CStr,
        memgraph: &M,
    ) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mgp_ptr = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeMemgraphStringValue,
                ffi::mgp_value_make_string,
                value.as_ptr(),
                mg.memory_ptr()
            )?;
            Ok(MgpValue::new(mgp_ptr, &mg))
        }
    }

    pub fn is_string(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_string, self.ptr).unwrap()
                != 0
        }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] object owns
    /// the new object.
    pub fn make_list<M: MemgraphMemory + ?Sized>(list: &List, memgraph: &M) -> Result<MgpValue> {
        fn to_local_error(_: Error) -> Error {
            Error::UnableToMakeListValue
        }
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mut list_copy = List::mgp_copy(list.mgp_ptr(), &mg).map_err(to_local_error)?;

            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeListValue,
                ffi::mgp_value_make_list,
                list_copy.mgp_ptr()
            )?;
            list_copy.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_list(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_list, self.ptr).unwrap() != 0
        }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] object owns
    /// the new object.
    pub fn make_map<M: MemgraphMemory + ?Sized>(map: &Map, memgraph: &M) -> Result<MgpValue> {
        fn to_local_error(_: Error) -> Error {
            Error::UnableToMakeMapValue
        }
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mut map_copy = Map::mgp_copy(map.mgp_ptr(), &mg).map_err(to_local_error)?;

            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeMapValue,
                ffi::mgp_value_make_map,
                map_copy.mgp_ptr()
            )?;
            map_copy.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_map(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_map, self.ptr).unwrap() != 0
        }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] object owns
    /// the new object.
    pub fn make_vertex<M: MemgraphMemory + ?Sized>(
        vertex: &Vertex,
        memgraph: &M,
    ) -> Result<MgpValue> {
        fn to_local_error(_: Error) -> Error {
            Error::UnableToMakeVertexValue
        }
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mut vertex_copy =
                Vertex::mgp_copy(vertex.mgp_ptr(), &mg).map_err(to_local_error)?;

            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeVertexValue,
                ffi::mgp_value_make_vertex,
                vertex_copy.mgp_ptr()
            )?;
            vertex_copy.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_vertex(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_vertex, self.ptr).unwrap()
                != 0
        }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] object owns
    /// the new object.
    pub fn make_edge<M: MemgraphMemory + ?Sized>(edge: &Edge, memgraph: &M) -> Result<MgpValue> {
        fn to_local_error(_: Error) -> Error {
            Error::UnableToMakeEdgeValue
        }
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mut edge_copy = Edge::mgp_copy(edge.mgp_ptr(), &mg).map_err(to_local_error)?;

            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeEdgeValue,
                ffi::mgp_value_make_edge,
                edge_copy.mgp_ptr()
            )?;
            edge_copy.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_edge(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_edge, self.ptr).unwrap() != 0
        }
    }

    /// Makes a copy of the given object returning the [MgpValue] object. [MgpValue] object owns
    /// the new object.
    pub fn make_path<M: MemgraphMemory + ?Sized>(path: &Path, memgraph: &M) -> Result<MgpValue> {
        fn to_local_error(_: Error) -> Error {
            Error::UnableToMakePathValue
        }
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        unsafe {
            let mut path_copy = Path::mgp_copy(path.mgp_ptr(), &mg).map_err(to_local_error)?;

            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakePathValue,
                ffi::mgp_value_make_path,
                path_copy.mgp_ptr()
            )?;
            path_copy.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_path(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_path, self.ptr).unwrap() != 0
        }
    }

    pub fn make_date<M: MemgraphMemory + ?Sized>(
        date: &NaiveDate,
        memgraph: &M,
    ) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        let mut date = temporal::Date::from_naive_date(date, &mg)?;
        unsafe {
            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeDateValue,
                ffi::mgp_value_make_date,
                date.mgp_ptr()
            )?;
            date.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_date(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_date, self.ptr).unwrap() != 0
        }
    }

    pub fn make_local_time<M: MemgraphMemory + ?Sized>(
        time: &NaiveTime,
        memgraph: &M,
    ) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        let mut local_time = temporal::LocalTime::from_naive_time(time, &mg)?;
        unsafe {
            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeLocalTimeValue,
                ffi::mgp_value_make_local_time,
                local_time.mgp_ptr()
            )?;
            local_time.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_local_time(&self) -> bool {
        unsafe {
            invoke_mgp_func!(
                ::std::os::raw::c_int,
                ffi::mgp_value_is_local_time,
                self.ptr
            )
            .unwrap()
                != 0
        }
    }

    pub fn make_local_date_time<M: MemgraphMemory + ?Sized>(
        datetime: &NaiveDateTime,
        memgraph: &M,
    ) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        let mut local_date_time = temporal::LocalDateTime::from_naive_date_time(datetime, &mg)?;
        unsafe {
            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeLocalDateTimeValue,
                ffi::mgp_value_make_local_date_time,
                local_date_time.mgp_ptr()
            )?;
            local_date_time.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_local_date_time(&self) -> bool {
        unsafe {
            invoke_mgp_func!(
                ::std::os::raw::c_int,
                ffi::mgp_value_is_local_date_time,
                self.ptr
            )
            .unwrap()
                != 0
        }
    }

    pub fn make_duration<M: MemgraphMemory + ?Sized>(
        duration: &chrono::Duration,
        memgraph: &M,
    ) -> Result<MgpValue> {
        let mg = memgraph.sealed_as_memgraph(MemorySeal(()));
        let mut duration = temporal::Duration::from_chrono_duration(duration, &mg)?;
        unsafe {
            let mgp_value = invoke_mgp_func_with_res!(
                *mut mgp_value,
                Error::UnableToMakeDurationValue,
                ffi::mgp_value_make_duration,
                duration.mgp_ptr()
            )?;
            duration.set_mgp_ptr(std::ptr::null_mut());
            Ok(MgpValue::new(mgp_value, &mg))
        }
    }

    pub fn is_duration(&self) -> bool {
        unsafe {
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_is_duration, self.ptr).unwrap()
                != 0
        }
    }
}

/// Object containing/owning concrete underlying mgp objects (e.g., mgp_vertex).
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(CString),
    Vertex(Vertex),
    Edge(Edge),
    Path(Path),
    List(List),
    Map(Map),
    Date(NaiveDate),
    LocalTime(NaiveTime),
    LocalDateTime(NaiveDateTime),
    Duration(chrono::Duration),
}

impl Value {
    pub fn to_mgp_value<M: MemgraphMemory + ?Sized>(&self, memgraph: &M) -> Result<MgpValue> {
        match self {
            Value::Null => MgpValue::make_null(memgraph),
            Value::Bool(x) => MgpValue::make_bool(*x, memgraph),
            Value::Int(x) => MgpValue::make_int(*x, memgraph),
            Value::Float(x) => MgpValue::make_double(*x, memgraph),
            Value::String(x) => MgpValue::make_string(x.as_c_str(), memgraph),
            Value::List(x) => MgpValue::make_list(x, memgraph),
            Value::Map(x) => MgpValue::make_map(x, memgraph),
            Value::Vertex(x) => MgpValue::make_vertex(x, memgraph),
            Value::Edge(x) => MgpValue::make_edge(x, memgraph),
            Value::Path(x) => MgpValue::make_path(x, memgraph),
            Value::Date(x) => MgpValue::make_date(x, memgraph),
            Value::LocalTime(x) => MgpValue::make_local_time(x, memgraph),
            Value::LocalDateTime(x) => MgpValue::make_local_date_time(x, memgraph),
            Value::Duration(x) => MgpValue::make_duration(x, memgraph),
        }
    }
}

impl From<MgpValue> for Value {
    fn from(item: MgpValue) -> Self {
        match item.to_value() {
            Ok(v) => v,
            Err(_) => panic!("Unable to create Value from MgpValue."),
        }
    }
}

/// Creates copy of [mgp_value] object as a [Value] object.
///
/// # Safety
///
/// Calls C API unsafe functions. The provided [mgp_value] object has to be a valid non-null
/// pointer.
pub(crate) unsafe fn mgp_raw_value_to_value(
    value: *mut mgp_value,
    memgraph: &Memgraph,
) -> Result<Value> {
    match invoke_mgp_func!(mgp_value_type, ffi::mgp_value_get_type, value).unwrap() {
        mgp_value_type::MGP_VALUE_TYPE_NULL => Ok(Value::Null),
        mgp_value_type::MGP_VALUE_TYPE_BOOL => Ok(Value::Bool(
            invoke_mgp_func!(::std::os::raw::c_int, ffi::mgp_value_get_bool, value).unwrap() != 0,
        )),
        mgp_value_type::MGP_VALUE_TYPE_INT => Ok(Value::Int(
            invoke_mgp_func!(i64, ffi::mgp_value_get_int, value).unwrap(),
        )),
        mgp_value_type::MGP_VALUE_TYPE_STRING => {
            let mgp_string =
                invoke_mgp_func!(*const c_char, ffi::mgp_value_get_string, value).unwrap();
            match create_cstring(mgp_string) {
                Ok(value) => Ok(Value::String(value)),
                Err(_) => Err(Error::UnableToMakeValueString),
            }
        }
        mgp_value_type::MGP_VALUE_TYPE_DOUBLE => Ok(Value::Float(
            invoke_mgp_func!(f64, ffi::mgp_value_get_double, value).unwrap(),
        )),
        mgp_value_type::MGP_VALUE_TYPE_VERTEX => Ok(Value::Vertex(Vertex::mgp_copy(
            invoke_mgp_func!(*mut mgp_vertex, ffi::mgp_value_get_vertex, value).unwrap(),
            memgraph,
        )?)),
        mgp_value_type::MGP_VALUE_TYPE_EDGE => Ok(Value::Edge(Edge::mgp_copy(
            invoke_mgp_func!(*mut mgp_edge, ffi::mgp_value_get_edge, value).unwrap(),
            memgraph,
        )?)),
        mgp_value_type::MGP_VALUE_TYPE_PATH => Ok(Value::Path(Path::mgp_copy(
            invoke_mgp_func!(*mut mgp_path, ffi::mgp_value_get_path, value).unwrap(),
            memgraph,
        )?)),
        mgp_value_type::MGP_VALUE_TYPE_LIST => Ok(Value::List(List::mgp_copy(
            invoke_mgp_func!(*mut mgp_list, ffi::mgp_value_get_list, value).unwrap(),
            memgraph,
        )?)),
        mgp_value_type::MGP_VALUE_TYPE_MAP => Ok(Value::Map(Map::mgp_copy(
            invoke_mgp_func!(*mut mgp_map, ffi::mgp_value_get_map, value).unwrap(),
            memgraph,
        )?)),
        mgp_value_type::MGP_VALUE_TYPE_DATE => Ok(Value::Date(
            temporal::Date::from_borrowed(
                invoke_mgp_func!(*mut mgp_date, ffi::mgp_value_get_date, value).unwrap(),
            )
            .to_naive_date(),
        )),
        mgp_value_type::MGP_VALUE_TYPE_LOCAL_TIME => Ok(Value::LocalTime(
            temporal::LocalTime::from_borrowed(
                invoke_mgp_func!(*mut mgp_local_time, ffi::mgp_value_get_local_time, value)
                    .unwrap(),
            )
            .to_naive_time(),
        )),
        mgp_value_type::MGP_VALUE_TYPE_LOCAL_DATE_TIME => Ok(Value::LocalDateTime(
            temporal::LocalDateTime::from_borrowed(
                invoke_mgp_func!(
                    *mut mgp_local_date_time,
                    ffi::mgp_value_get_local_date_time,
                    value
                )
                .unwrap(),
            )
            .to_naive_date_time(),
        )),
        mgp_value_type::MGP_VALUE_TYPE_DURATION => Ok(Value::Duration(
            temporal::Duration::from_borrowed(
                invoke_mgp_func!(*mut mgp_duration, ffi::mgp_value_get_duration, value).unwrap(),
            )
            .to_chrono_duration(),
        )),
        mgp_value_type::MGP_VALUE_TYPE_ZONED_DATE_TIME
        | mgp_value_type::MGP_VALUE_TYPE_POINT_2D
        | mgp_value_type::MGP_VALUE_TYPE_POINT_3D
        | mgp_value_type::MGP_VALUE_TYPE_ENUM => Err(Error::UnsupportedValueType),
    }
}

#[cfg(test)]
mod tests;
