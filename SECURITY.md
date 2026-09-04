# Security Policy

This software is verified for security for official releases and as such does
not make promises about the quality of the product for patches delivered between
releases.

## Security Boundaries

The ML Workload Library for Vulkan® is not a sandbox or process-security boundary.
Data supplied through its public API are expected to come from
trusted sources. The library also relies on the operating system, filesystem,
caller, Vulkan® Loader, and installable client driver (ICD) being trusted.
Weaknesses originating in those components or in a compromised same-process
caller, cannot be addressed in this library.

## Reporting a Vulnerability

Security vulnerabilities may be reported to the Arm® Product Security Incident
Response Team (PSIRT) by sending an email to
[psirt@arm.com](mailto:psirt@arm.com).

For more information visit
<https://developer.arm.com/support/arm-security-updates/report-security-vulnerabilities>
