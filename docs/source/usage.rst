Using the ML Workload Library for Vulkan®
==========================================

Execution model
---------------

The library can create a runtime-owned Vulkan® context, or it can wrap Vulkan®
objects supplied by the application without taking ownership. When wrapping
objects, the application must enable the Vulkan® extensions and features required
by the workload. Similarly, the application can supply resources, or the context
can allocate them based on the workload resource requirements.

A workload is executed through the following sequence:

1. Construct a ``Workload``.
2. Create a runtime-owned ``Context``, or wrap the application's Vulkan® instance,
   physical device, device, queue family, and queue.

3. Create a ``Session``, bind any needed placeholder module or data-graph
   binary, then call ``Session::configure()``.

   - For a placeholder module, call
     ``Session::bindModule(workload.placeholderModule(index), implementation)``.
   - For a standalone data graph backed by a binary, call
     ``Session::bindDataGraphBinary(binary)``.

   These compilation inputs are alternatives; a Session cannot bind both.

4. Create a ``BindingSet`` and bind each public resource using caller-owned or
   runtime-owned allocations.
5. Prepare a binding snapshot and either run it or record it into an
   application command buffer.

By default, each ``Session`` owns its compiled pipelines and layouts. To share
them across sessions for the same context and workload, pass a
``CompiledExecution`` to each session. Each session still has its own binding
sets and command state.

Each session also has its own tiling selection. Building shared pipelines fixes
graph tensor tiling and the tiling of its alias peers. Sessions created
afterward start with those choices; other tiling choices remain independent. A
session created earlier must select compatible graph tensor tiling before it
can reuse the pipelines.

The following excerpt from the standalone compute sample shows the common flow
with a runtime-owned context and buffers. The sample defines the GLSL workload
description and mapped-memory helpers earlier in the source file.

.. literalinclude:: ../sources/samples/2_run_glsl_compute.cpp
   :language: cpp
   :start-after: // [standalone-compute-execution-begin]
   :end-before: // [standalone-compute-execution-end]
   :dedent: 8

An application can instead retain ownership of its Vulkan® objects and wrap
them in a library context:

.. literalinclude:: ../sources/samples/4_record_glsl_compute_with_wrapped_context.cpp
   :language: cpp
   :start-after: // [wrapped-context-begin]
   :end-before: // [wrapped-context-end]
   :dedent: 8

Complete programs showing workload inspection, VGF and standalone compute
execution, and application-owned Vulkan® context wrapping are available in the
``samples`` directory.

Workload sources
----------------

``Workload::fromVGF(...)`` loads a VGF file or decodes a caller-owned memory
buffer. ``Workload::fromComputeShader(...)`` constructs a standalone compute
workload. ``Workload::fromDataGraph(...)`` constructs a standalone Vulkan®
data-graph workload.

The standalone compute samples describe their module, dispatch, and resources
directly:

.. literalinclude:: ../sources/samples/sample_utils.hpp
   :language: cpp
   :start-after: // [compute-description-begin]
   :end-before: // [compute-description-end]

Standalone data-graph workloads use the same resource model and add graph
pipeline metadata. Sample 5 leaves its module implementation unspecified so
the Session can bind a module or binary later:

.. literalinclude:: ../sources/samples/5_create_standalone_data_graph.cpp
   :language: cpp
   :start-after: // [data-graph-description-begin]
   :end-before: // [data-graph-description-end]
   :dedent: 8

Modules can contain SPIR-V™, or GLSL and HLSL source when the matching optional
backend is available. GLSL and HLSL modules are supported for compute
executables. Check an optional backend with ``supports(Feature::GlslModules)``
or ``supports(Feature::HlslModules)``.

.. literalinclude:: ../sources/samples/2_run_glsl_compute.cpp
   :language: cpp
   :start-after: // [glsl-support-check-begin]
   :end-before: // [glsl-support-check-end]
   :dedent: 8

A workload can also contain placeholder modules. Sample 3 binds a GLSL
implementation before configuring its session:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [placeholder-module-binding-begin]
   :end-before: // [placeholder-module-binding-end]
   :dedent: 8

The VGF execution sample encodes a small in-memory VGF with a placeholder
module:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [vgf-building-begin]
   :end-before: // [vgf-building-end]

The encoded bytes remain alive while ``Workload`` decodes their caller-owned
memory:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [in-memory-vgf-loading-begin]
   :end-before: // [in-memory-vgf-loading-end]
   :dedent: 8

Workload inspection
-------------------

A workload exposes non-owning views of its public resources. Sample 1 inspects
the identity, access, kind, and kind-specific requirements of each resource:

.. literalinclude:: ../sources/samples/1_inspect_workload_metadata.cpp
   :language: cpp
   :start-after: // [resource-inspection-begin]
   :end-before: // [resource-inspection-end]

Similarly, executables expose their module and descriptor interface:

.. literalinclude:: ../sources/samples/1_inspect_workload_metadata.cpp
   :language: cpp
   :start-after: // [executable-inspection-begin]
   :end-before: // [executable-inspection-end]

Resource binding
----------------

The workload exposes its public resources in order through
``Workload::resources()`` or by index through ``Workload::resource(...)``.

Choose public tensor or image tiling with ``Session::setTensorTiling()`` or
``Session::setImageTiling()`` before calling ``configure()``. Without a
selection, tensors use Linear tiling and images use Optimal tiling. A tensor
that requires image aliasing uses Optimal. Alias groups use a common default
tiling: Optimal if any peer defaults to Optimal, unless a storage-buffer peer
or explicit tensor strides require Linear. Image/tensor alias groups are
rejected if a tensor peer's innermost dimension exceeds four, regardless of
tiling.
Incompatible alias tiling requirements are rejected. For Optimal image/tensor
aliases, the required tensor usage includes ``eImageAliasing``.

The allocation overloads without a tiling argument create Linear tensors and
Optimal images. ``ImageRequirementsView::usage()`` returns the usage for
Optimal tiling.

To create and bind resources:

#. Create each public resource from its kind and requirements, or use
   ``Context::createTensor()``, ``Context::createBuffer()``, or
   ``Context::createImage()``. For tensors and images, use the tiling returned
   by ``session.tensorTiling(resource)`` or ``session.imageTiling(resource)``
   when creating the Vulkan® object; pass it to the ``Context`` helper if
   using one. For a tensor created directly, query its required usage with
   ``resource.requirements().asTensor().usage(session.tensorTiling(resource))``.
   For an image created directly, query its required usage with
   ``resource.requirements().asImage().usage(session.imageTiling(resource))``.

#. Bind every required public resource with ``bindTensor()``, ``bindBuffer()``,
   or ``bindImage()`` before preparing an execution.

``BindingSet`` cannot inspect a caller-owned Vulkan® handle's creation tiling.

Sample 3 binds every public resource according to its kind:

.. literalinclude:: ../sources/samples/3_run_vgf_with_placeholder_module.cpp
   :language: cpp
   :start-after: // [runtime-resource-binding-begin]
   :end-before: // [runtime-resource-binding-end]

Supply ``BoundMemoryInfo`` when
``ResourceRequirementsView::requiresBoundMemoryInfo()`` is true. Image bindings
must provide an image view and the required subresource range, and must provide a
sampler only when the image requirements request one.

An image layout is optional. When supplied, it describes the current layout, and
prepared execution transitions the image to the workload-required layout. When
omitted, the application manages the image layout.

Running and recording
---------------------

``PreparedExecution::run()`` records, submits, and waits for completion using
library-managed command and fence state.

``PreparedExecution::record()`` records into a caller-owned command buffer. The
application is responsible for the command-buffer lifecycle, submission, and
synchronization.

.. literalinclude:: ../sources/samples/4_record_glsl_compute_with_wrapped_context.cpp
   :language: cpp
   :start-after: // [recorded-execution-begin]
   :end-before: // [recorded-execution-end]
   :dedent: 8

Pass ``RecordOptions`` to the ``record()`` overload to record caller-owned
commands immediately before and after each executable dispatch. The hook
receives the recording command buffer and an ``ExecutableView`` whose index,
name, and kind identify the executable, together with an
``ExecutableRecordPhase`` indicating whether it runs before or after the
dispatch. This supports timestamps and debug labels without transferring
ownership of their Vulkan® objects to the library.

Pipeline options
----------------

``SessionOptions`` accepts a pipeline cache and optional compute-pipeline
robustness and neural statistics settings. The application must enable the
corresponding Vulkan® extensions and device features before requesting them.
Neural statistics settings are applied during graph pipeline and graph session
creation.
The pipeline cache is not used for binary-backed pipeline creation.

Ownership and lifetimes
-----------------------

- The application retains ownership of Vulkan® objects and resources it supplies.
  Runtime allocations own the Vulkan® resource and backing memory they expose.
- All bound resources, whether caller-owned or runtime-owned, must remain valid
  until submitted work has completed.
- A runtime-owned ``Context`` must outlive its sessions, allocations, and any
  submitted work that uses them.
- The Vulkan® objects wrapped by ``Context`` must outlive the context, its
  sessions, allocations created through it, and submitted work that uses them.
- A ``Workload`` must outlive its non-owning views and every ``Session`` created
  from it. A ``Session`` must outlive its binding sets and prepared executions.
- ``CompiledExecution`` is tied to the ``Context`` and ``Workload`` supplied at
  creation. Sessions keep its compiled state alive if the ``CompiledExecution``
  object is destroyed; the context and workload must still outlive those
  sessions.
- VGF memory passed to ``Workload::fromVGF(data, size)`` must remain valid for
  the workload lifetime. The same applies to caller-owned constant payloads in
  a standalone data-graph description.
- The library borrows data-graph binary bytes when they are bound to a
  ``Session``. The pointer must meet the device's
  ``dataGraphPipelineBinaryDataAlignment`` requirement. With the default
  ``DataGraphBinaryStorageMode::ImplementationManaged`` mode, keep the bytes
  valid until ``Session::configure()`` returns. If pipeline creation fails and
  is retried, keep them alive or bind a replacement binary before retrying. The optional
  identifier is copied when bound.
- With ``DataGraphBinaryStorageMode::ApplicationBacked``, Vulkan® may use the
  supplied bytes directly. Keep them accessible and unchanged until every
  pipeline created from them is destroyed, including compiled state shared by Sessions.
- ``Session::prepare()`` snapshots a binding set. Later changes to that binding
  set do not change the prepared execution.
