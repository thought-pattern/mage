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
//! Abstraction to interact with Memgraph.

use std::cell::Cell;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::rc::Rc;

use crate::list::*;
use crate::mgp::*;
use crate::result::*;
use crate::rsmgp::*;
use crate::vertex::*;
// Required here, if not present, tests linking fails.
#[double]
use crate::mgp::ffi;
use mockall_double::double;

#[derive(Debug, PartialEq)]
pub enum MgpError {
    UnknownError,
    UnableToAllocate,
    InsufficientError,
    OutOfRange,
    LogicError,
    DeletedObject,
    InvalidArgument,
    KeyAlreadyExists,
    ImmutableObject,
    ValueConversion,
    SerializationError,
    AuthorizationError,
    NotYetImplemented,
}

pub(crate) trait MgpDefault {
    fn default() -> Self;
}

macro_rules! mgp_default_mut_ptr {
    ($t:ident) => {
        impl MgpDefault for *mut $t {
            fn default() -> Self {
                std::ptr::null_mut()
            }
        }
    };
}

macro_rules! mgp_default_const_ptr {
    ($t:ident) => {
        impl MgpDefault for *const $t {
            fn default() -> Self {
                std::ptr::null()
            }
        }
    };
}

macro_rules! mgp_default_zero {
    ($t:ident) => {
        impl MgpDefault for $t {
            fn default() -> Self {
                0
            }
        }
    };
}

mgp_default_mut_ptr!(mgp_date);
mgp_default_mut_ptr!(mgp_duration);
mgp_default_mut_ptr!(mgp_edge);
mgp_default_mut_ptr!(mgp_edges_iterator);
mgp_default_mut_ptr!(mgp_list);
mgp_default_mut_ptr!(mgp_local_date_time);
mgp_default_mut_ptr!(mgp_local_time);
mgp_default_mut_ptr!(mgp_map);
mgp_default_mut_ptr!(mgp_map_item);
mgp_default_mut_ptr!(mgp_map_items_iterator);
mgp_default_mut_ptr!(mgp_path);
mgp_default_mut_ptr!(mgp_proc);
mgp_default_mut_ptr!(mgp_properties_iterator);
mgp_default_mut_ptr!(mgp_property);
mgp_default_mut_ptr!(mgp_result_record);
mgp_default_mut_ptr!(mgp_type);
mgp_default_mut_ptr!(mgp_value);
mgp_default_mut_ptr!(mgp_vertex);
mgp_default_mut_ptr!(mgp_vertices_iterator);

mgp_default_const_ptr!(c_char);
mgp_default_const_ptr!(u64);

mgp_default_zero!(i32);
mgp_default_zero!(i64);
mgp_default_zero!(u64);
mgp_default_zero!(usize);

impl MgpDefault for mgp_edge_id {
    fn default() -> Self {
        mgp_edge_id { as_int: 0 }
    }
}

impl MgpDefault for mgp_vertex_id {
    fn default() -> Self {
        mgp_vertex_id { as_int: 0 }
    }
}

impl MgpDefault for mgp_edge_type {
    fn default() -> Self {
        mgp_edge_type {
            name: std::ptr::null(),
        }
    }
}

impl MgpDefault for mgp_label {
    fn default() -> Self {
        mgp_label {
            name: std::ptr::null(),
        }
    }
}

impl MgpDefault for mgp_value_type {
    fn default() -> Self {
        mgp_value_type::MGP_VALUE_TYPE_NULL
    }
}

impl MgpDefault for f64 {
    fn default() -> Self {
        0.0
    }
}

pub(crate) fn to_rust_mgp_error(error: mgp_error) -> Option<MgpError> {
    match error {
        mgp_error::MGP_ERROR_NO_ERROR => None,
        mgp_error::MGP_ERROR_UNKNOWN_ERROR => Some(MgpError::UnknownError),
        mgp_error::MGP_ERROR_UNABLE_TO_ALLOCATE => Some(MgpError::UnableToAllocate),
        mgp_error::MGP_ERROR_INSUFFICIENT_BUFFER => Some(MgpError::InsufficientError),
        mgp_error::MGP_ERROR_OUT_OF_RANGE => Some(MgpError::OutOfRange),
        mgp_error::MGP_ERROR_LOGIC_ERROR => Some(MgpError::LogicError),
        mgp_error::MGP_ERROR_DELETED_OBJECT => Some(MgpError::DeletedObject),
        mgp_error::MGP_ERROR_INVALID_ARGUMENT => Some(MgpError::InvalidArgument),
        mgp_error::MGP_ERROR_KEY_ALREADY_EXISTS => Some(MgpError::KeyAlreadyExists),
        mgp_error::MGP_ERROR_IMMUTABLE_OBJECT => Some(MgpError::ImmutableObject),
        mgp_error::MGP_ERROR_VALUE_CONVERSION => Some(MgpError::ValueConversion),
        mgp_error::MGP_ERROR_SERIALIZATION_ERROR => Some(MgpError::SerializationError),
        mgp_error::MGP_ERROR_AUTHORIZATION_ERROR => Some(MgpError::AuthorizationError),
        mgp_error::MGP_ERROR_NOT_YET_IMPLEMENTED => Some(MgpError::NotYetImplemented),
    }
}

macro_rules! invoke_mgp_func {
    ($result_type:ty, $func:expr) => {{
        let result : Result<$result_type, MgpError> = {
            let mut result: $result_type = MgpDefault::default();
            match to_rust_mgp_error($func(&mut result)) {
                None => Ok(result),
                Some(err) => Err(err),
            }
        };
        result
    }};
    ($result_type:ty, $func:expr, $($args:expr),+) => {{
        let result : Result<$result_type, MgpError> = {
            let mut result: $result_type = MgpDefault::default();
            match to_rust_mgp_error($func($($args),+, &mut result )) {
                None => Ok(result),
                Some(err) => Err(err),
            }
        };
        result
    }};
}

macro_rules! invoke_void_mgp_func {
    ($func:expr, $($args:expr),+) => {{
        let result : Result<(), MgpError> = {
            match to_rust_mgp_error($func($($args),+)) {
                None => Ok(()),
                Some(err) => Err(err),
            }
        };
        result
    }};
}

macro_rules! invoke_mgp_func_with_res {
    ($result_type:ty, $err:expr, $func:expr) => {{
        invoke_mgp_func!($result_type, $func).map_err(|_| -> Error { $err })
    }};
    ($result_type:ty, $err:expr, $func:expr, $($args:expr),+) => {{
        invoke_mgp_func!($result_type, $func, $($args),+).map_err(|_| -> Error { $err })
    }};
}

macro_rules! invoke_void_mgp_func_with_res {
    ($err:expr, $func:expr, $($args:expr),+) => {{
        invoke_void_mgp_func!($func, $($args),+).map_err(|_| -> Error { $err })
    }};
}

pub(crate) use invoke_mgp_func;
pub(crate) use invoke_mgp_func_with_res;
pub(crate) use invoke_void_mgp_func;
pub(crate) use invoke_void_mgp_func_with_res;

/// Combines the given array of types from left to right to construct [mgp_type]. E.g., if the
/// input is [Type::List, Type::Int], the constructed [mgp_type] is going to be list of integers.
fn resolve_mgp_type(types: &[Type]) -> *mut mgp_type {
    unsafe {
        let mut mgp_type_ptr: *mut mgp_type = std::ptr::null_mut();
        for field_type in types.iter().rev() {
            mgp_type_ptr = match field_type {
                Type::Any => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_any).unwrap(),
                Type::Bool => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_bool).unwrap(),
                Type::Number => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_number).unwrap(),
                Type::Int => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_int).unwrap(),
                Type::Double => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_float).unwrap(),
                Type::String => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_string).unwrap(),
                Type::Map => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_map).unwrap(),
                Type::Vertex => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_node).unwrap(),
                Type::Edge => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_relationship).unwrap(),
                Type::Path => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_path).unwrap(),
                Type::Nullable => {
                    invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_nullable, mgp_type_ptr).unwrap()
                }
                Type::List => {
                    invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_list, mgp_type_ptr).unwrap()
                }
                Type::Date => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_date).unwrap(),
                Type::LocalTime => {
                    invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_local_time).unwrap()
                }
                Type::LocalDateTime => {
                    invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_local_date_time).unwrap()
                }
                Type::Duration => invoke_mgp_func!(*mut mgp_type, ffi::mgp_type_duration).unwrap(),
            };
        }
        mgp_type_ptr
    }
}

unsafe fn configure_procedure(
    procedure: *mut mgp_proc,
    required_arg_types: &[NamedType],
    optional_arg_types: &[OptionalNamedType],
    result_field_types: &[NamedType],
) -> Result<()> {
    for required_type in required_arg_types {
        let mgp_type = resolve_mgp_type(required_type.types);
        if ffi::mgp_proc_add_arg(procedure, required_type.name.as_ptr(), mgp_type)
            != mgp_error::MGP_ERROR_NO_ERROR
        {
            return Err(Error::UnableToAddRequiredArguments);
        }
    }

    for optional_input in optional_arg_types {
        let mgp_type = resolve_mgp_type(optional_input.types);

        if ffi::mgp_proc_add_opt_arg(
            procedure,
            optional_input.name.as_ptr(),
            mgp_type,
            optional_input.default.mgp_ptr(),
        ) != mgp_error::MGP_ERROR_NO_ERROR
        {
            return Err(Error::UnableToAddOptionalArguments);
        }
    }

    for result_field in result_field_types {
        let mgp_type = resolve_mgp_type(result_field.types);
        if result_field.deprecated {
            if ffi::mgp_proc_add_deprecated_result(
                procedure,
                result_field.name.as_ptr(),
                mgp_type,
            ) != mgp_error::MGP_ERROR_NO_ERROR
            {
                return Err(Error::UnableToAddDeprecatedReturnType);
            }
        } else if ffi::mgp_proc_add_result(procedure, result_field.name.as_ptr(), mgp_type)
            != mgp_error::MGP_ERROR_NO_ERROR
        {
            return Err(Error::UnableToAddReturnType);
        }
    }

    Ok(())
}

/// Crate-private seal token preventing external implementations or direct calls on
/// [MemgraphMemory] internal methods.
#[doc(hidden)]
pub struct MemorySeal(pub(crate) ());

/// Trait implemented by callback contexts ([Memgraph] and [MemgraphModule]) that own an
/// [mgp_memory] arena.
pub trait MemgraphMemory {
    #[doc(hidden)]
    fn sealed_memory_ptr(&self, _seal: MemorySeal) -> *mut mgp_memory;

    #[doc(hidden)]
    fn sealed_as_memgraph(&self, _seal: MemorySeal) -> Memgraph;
}

/// Procedure-execution handle exposing the arguments, graph, result, and memory valid during a
/// procedure call.
#[derive(Clone)]
pub struct Memgraph {
    args: *mut mgp_list,
    graph: *mut mgp_graph,
    result: *mut mgp_result,
    memory: *mut mgp_memory,
    record_generation: Rc<Cell<u64>>,
}

/// Module-initialization handle exposing procedure registration and default-argument memory
/// allocation valid during `mgp_init_module`.
#[derive(Clone)]
pub struct MemgraphModule {
    module: *mut mgp_module,
    memory: *mut mgp_memory,
}

impl MemgraphMemory for Memgraph {
    fn sealed_memory_ptr(&self, _seal: MemorySeal) -> *mut mgp_memory {
        self.memory
    }

    fn sealed_as_memgraph(&self, _seal: MemorySeal) -> Memgraph {
        self.clone()
    }
}

impl MemgraphMemory for MemgraphModule {
    fn sealed_memory_ptr(&self, _seal: MemorySeal) -> *mut mgp_memory {
        self.memory
    }

    fn sealed_as_memgraph(&self, _seal: MemorySeal) -> Memgraph {
        Memgraph::from_memory(self.memory)
    }
}

impl<T: MemgraphMemory + ?Sized> MemgraphMemory for &T {
    fn sealed_memory_ptr(&self, seal: MemorySeal) -> *mut mgp_memory {
        (**self).sealed_memory_ptr(seal)
    }

    fn sealed_as_memgraph(&self, seal: MemorySeal) -> Memgraph {
        (**self).sealed_as_memgraph(seal)
    }
}

impl Memgraph {
    /// Creates a procedure-execution [Memgraph] context from raw callback pointers.
    ///
    /// # Safety
    ///
    /// The pointers must come from the active Memgraph procedure invocation.
    #[doc(hidden)]
    pub unsafe fn new(
        args: *mut mgp_list,
        graph: *mut mgp_graph,
        result: *mut mgp_result,
        memory: *mut mgp_memory,
    ) -> Memgraph {
        Memgraph {
            args,
            graph,
            result,
            memory,
            record_generation: Rc::new(Cell::new(0)),
        }
    }

    pub(crate) fn from_memory(memory: *mut mgp_memory) -> Memgraph {
        Memgraph {
            args: std::ptr::null_mut(),
            graph: std::ptr::null_mut(),
            result: std::ptr::null_mut(),
            memory,
            record_generation: Rc::new(Cell::new(0)),
        }
    }

    /// Creates a new object with all underlying data set to null. Used for testing purposes.
    #[cfg(test)]
    pub(crate) fn new_default() -> Memgraph {
        Memgraph {
            args: std::ptr::null_mut(),
            graph: std::ptr::null_mut(),
            result: std::ptr::null_mut(),
            memory: std::ptr::null_mut(),
            record_generation: Rc::new(Cell::new(0)),
        }
    }

    pub(crate) fn next_record_generation(&self) -> Option<u64> {
        let next = self.record_generation.get().checked_add(1)?;
        self.record_generation.set(next);
        Some(next)
    }

    pub(crate) fn is_active_record_generation(&self, generation: u64) -> bool {
        generation != 0 && self.record_generation.get() == generation
    }

    /// Arguments passed to the procedure call.
    pub fn args(&self) -> Result<List> {
        #[cfg(not(test))]
        if self.args.is_null() {
            return Err(Error::UnableToCopyList);
        }
        unsafe { List::mgp_copy(self.args_ptr(), self) }
    }

    /// Returns pointer to the object with all arguments passed to the procedure call.
    pub(crate) fn args_ptr(&self) -> *mut mgp_list {
        self.args
    }

    /// Returns pointer to the object with graph data.
    pub(crate) fn graph_ptr(&self) -> *mut mgp_graph {
        self.graph
    }

    /// Returns pointer to the object where results could be stored.
    pub(crate) fn result_ptr(&self) -> *mut mgp_result {
        self.result
    }

    /// Returns pointer to the memory object for advanced memory control.
    pub(crate) fn memory_ptr(&self) -> *mut mgp_memory {
        self.memory
    }

    pub fn vertices_iter(&self) -> Result<VerticesIterator> {
        #[cfg(not(test))]
        if self.graph.is_null() {
            return Err(Error::UnableToCreateGraphVerticesIterator);
        }
        unsafe {
            let mgp_iterator = invoke_mgp_func!(
                *mut mgp_vertices_iterator,
                ffi::mgp_graph_iter_vertices,
                self.graph_ptr(),
                self.memory_ptr()
            )
            .map_err(|_| Error::UnableToCreateGraphVerticesIterator)?;
            if mgp_iterator.is_null() {
                return Err(Error::UnableToCreateGraphVerticesIterator);
            }
            Ok(VerticesIterator::new(mgp_iterator, self))
        }
    }

    pub fn vertex_by_id(&self, id: i64) -> Result<Vertex> {
        #[cfg(not(test))]
        if self.graph.is_null() {
            return Err(Error::UnableToFindVertexById);
        }
        unsafe {
            let mgp_vertex_ptr = invoke_mgp_func!(
                *mut mgp_vertex,
                ffi::mgp_graph_get_vertex_by_id,
                self.graph_ptr(),
                mgp_vertex_id { as_int: id },
                self.memory_ptr()
            )
            .map_err(|_| Error::UnableToFindVertexById)?;
            if mgp_vertex_ptr.is_null() {
                return Err(Error::UnableToFindVertexById);
            }
            Ok(Vertex::new(mgp_vertex_ptr, self))
        }
    }

    /// Creates a new result record.
    ///
    /// Creating a new record invalidates any previously created [ResultRecord] on the same query
    /// result across all handles.
    pub fn result_record(&self) -> Result<ResultRecord> {
        ResultRecord::create(self)
    }

    /// Return `true` if the currently executing procedure should abort as soon as possible.
    pub fn must_abort(&self) -> bool {
        if self.graph.is_null() {
            return false;
        }
        unsafe { ffi::mgp_must_abort(self.graph_ptr()) != 0 }
    }
}

impl MemgraphModule {
    /// Creates a module-initialization [MemgraphModule] context from raw callback pointers.
    ///
    /// # Safety
    ///
    /// The pointers must come from the active `mgp_init_module` invocation.
    #[doc(hidden)]
    pub unsafe fn new(module: *mut mgp_module, memory: *mut mgp_memory) -> MemgraphModule {
        MemgraphModule { module, memory }
    }

    #[cfg(test)]
    pub(crate) fn new_default() -> MemgraphModule {
        MemgraphModule {
            module: std::ptr::null_mut(),
            memory: std::ptr::null_mut(),
        }
    }

    /// Returns pointer to the module object.
    pub fn module_ptr(&self) -> *mut mgp_module {
        self.module
    }

    /// Returns pointer to the memory object.
    pub(crate) fn memory_ptr(&self) -> *mut mgp_memory {
        self.memory
    }

    /// Registers a new read procedure.
    pub fn add_read_procedure(
        &self,
        proc_ptr: extern "C" fn(*mut mgp_list, *mut mgp_graph, *mut mgp_result, *mut mgp_memory),
        name: &CStr,
        required_arg_types: &[NamedType],
        optional_arg_types: &[OptionalNamedType],
        result_field_types: &[NamedType],
    ) -> Result<()> {
        #[cfg(not(test))]
        if self.module.is_null() {
            return Err(Error::UnableToRegisterReadProcedure);
        }
        unsafe {
            let procedure = invoke_mgp_func!(
                *mut mgp_proc,
                ffi::mgp_module_add_read_procedure,
                self.module_ptr(),
                name.as_ptr(),
                Some(proc_ptr)
            )
            .map_err(|_| Error::UnableToRegisterReadProcedure)?;
            configure_procedure(
                procedure,
                required_arg_types,
                optional_arg_types,
                result_field_types,
            )
        }
    }

    /// Registers a new write procedure.
    pub fn add_write_procedure(
        &self,
        proc_ptr: extern "C" fn(*mut mgp_list, *mut mgp_graph, *mut mgp_result, *mut mgp_memory),
        name: &CStr,
        required_arg_types: &[NamedType],
        optional_arg_types: &[OptionalNamedType],
        result_field_types: &[NamedType],
    ) -> Result<()> {
        #[cfg(not(test))]
        if self.module.is_null() {
            return Err(Error::UnableToRegisterWriteProcedure);
        }
        unsafe {
            let procedure = invoke_mgp_func!(
                *mut mgp_proc,
                ffi::mgp_module_add_write_procedure,
                self.module_ptr(),
                name.as_ptr(),
                Some(proc_ptr)
            )
            .map_err(|_| Error::UnableToRegisterWriteProcedure)?;
            configure_procedure(
                procedure,
                required_arg_types,
                optional_arg_types,
                result_field_types,
            )
        }
    }
}

#[cfg(test)]
mod tests;
