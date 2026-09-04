Using the ML Workload Library for Vulkan®
==========================================

Execution model
---------------

The library can create a runtime-owned Vulkan® context, or it can wrap Vulkan®
objects supplied by the application without taking ownership. When wrapping
objects, the application must enable the Vulkan® extensions and features required
by the workload. Resources can likewise be supplied by the application or
allocated through the context from workload resource requirements.

A workload is executed through the following sequence:

1. Construct a ``Workload``.
2. Create a runtime-owned ``Context``, or wrap the application's Vulkan® instance,
   physical device, device, queue family, and queue.
3. Create a ``Session``, provide any missing module implementations, and
   configure it.
4. Create a ``BindingSet`` and bind each public resource using caller-owned or
   runtime-owned allocations.
5. Prepare a binding snapshot and either run it or record it into an
   application command buffer.

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

.. literalinclude:: ../sources/samples/4_run_glsl_compute_with_wrapped_context.cpp
   :language: cpp
   :start-after: // [wrapped-context-begin]
   :end-before: // [wrapped-context-end]
   :dedent: 8

Complete programs showing VGF inspection and execution, standalone compute
execution, and application-owned Vulkan® context wrapping are available in the
``samples`` directory.

Workload sources
----------------

``Workload::fromVGF(...)`` loads a VGF file or decodes a caller-owned memory
buffer. ``Workload::fromComputeShader(...)`` constructs a standalone compute
workload, and ``Workload::fromDataGraph(...)`` constructs a standalone Vulkan®
data-graph workload.

The standalone compute samples describe their module, dispatch, and resources
directly:

.. literalinclude:: ../sources/samples/sample_utils.hpp
   :language: cpp
   :start-after: // [compute-description-begin]
   :end-before: // [compute-description-end]

Standalone data-graph workloads use the same resource model and add graph
pipeline metadata. Sample 5 leaves the module implementation missing so it can
be supplied to a session later:

.. literalinclude:: ../sources/samples/5_create_data_graph.cpp
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

.. literalinclude:: ../sources/samples/3_run_vgf.cpp
   :language: cpp
   :start-after: // [placeholder-module-binding-begin]
   :end-before: // [placeholder-module-binding-end]
   :dedent: 8

The VGF samples use a shared helper to encode an in-memory VGF:

.. literalinclude:: ../sources/samples/sample_utils.hpp
   :language: cpp
   :start-after: // [vgf-building-begin]
   :end-before: // [vgf-building-end]

The encoded bytes remain alive while ``Workload`` decodes their caller-owned
memory:

.. literalinclude:: ../sources/samples/1_inspect_vgf.cpp
   :language: cpp
   :start-after: // [in-memory-vgf-loading-begin]
   :end-before: // [in-memory-vgf-loading-end]
   :dedent: 8

Workload inspection
-------------------

A workload exposes non-owning views of its public resources. Sample 1 inspects
each resource's identity, access, kind, and kind-specific requirements:

.. literalinclude:: ../sources/samples/1_inspect_vgf.cpp
   :language: cpp
   :start-after: // [resource-inspection-begin]
   :end-before: // [resource-inspection-end]

Executables similarly expose their module and descriptor interface:

.. literalinclude:: ../sources/samples/1_inspect_vgf.cpp
   :language: cpp
   :start-after: // [executable-inspection-begin]
   :end-before: // [executable-inspection-end]

Resource binding
----------------

The workload exposes its public resources in order through
``Workload::resources()`` or by index through ``Workload::resource(...)``.
Inspect each resource's kind and requirements before creating the corresponding
Vulkan® object, or use ``Context::createTensor()``, ``Context::createBuffer()``,
or ``Context::createImage()`` to create a compatible allocation. Then use
``bindTensor()``, ``bindBuffer()``, or ``bindImage()``. Every required public
resource must be bound before preparing an execution.

Sample 3 applies this process to every public resource according to its kind:

.. literalinclude:: ../sources/samples/3_run_vgf.cpp
   :language: cpp
   :start-after: // [runtime-resource-binding-begin]
   :end-before: // [runtime-resource-binding-end]

Supply ``BoundMemoryInfo`` when
``ResourceRequirementsView::requiresBoundMemoryInfo()`` is true. Image bindings
must provide an image view and the required subresource range, and must provide a
sampler only when the image requirements request one. An image layout is
optional: when supplied, it describes the current layout and prepared execution
transitions the image to the workload-required layout. When omitted, the
application manages the image layout.

Running and recording
---------------------

``PreparedExecution::run()`` records, submits, and waits for completion using
library-managed command and fence state.

``PreparedExecution::record()`` records into a caller-owned command buffer. The
application is responsible for the command-buffer lifecycle, submission, and
synchronization.

.. literalinclude:: ../sources/samples/4_run_glsl_compute_with_wrapped_context.cpp
   :language: cpp
   :start-after: // [recorded-execution-begin]
   :end-before: // [recorded-execution-end]
   :dedent: 8

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
- VGF memory passed to ``Workload::fromVGF(data, size)`` must remain valid for
  the workload lifetime. The same applies to caller-owned constant payloads in
  a standalone data-graph description.
- ``Session::prepare()`` snapshots a binding set. Later changes to that binding
  set do not change the prepared execution.
