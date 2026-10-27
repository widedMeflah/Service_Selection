"""
Prompts of the Documentation-Grounded Completion module. Same structure as every LLM call of the approach:
Role (i), Context (ii), Task (iii), Instructions (iv), Output format (v).

EXTRACTION_PROMPT   Fig. 3: all the dependent resources of a service in one chunk of an official page
URL_PROMPT          candidate URLs of official pages, when no URL is known for a service
REPAIR_PROMPT       configures again a resource that fails a verification, so as to satisfy the policy
"""

EXTRACTION_PROMPT = """Role (i)
You are an expert in cloud infrastructure and provider documentation.

Context (ii)
Considering the following service and official documentation page:
Service: {service}
Provider: {provider}
Page: {page}

Task (iii)
Identify, in this page, all the dependent resources of the service, i.e., the resources it requires from its provider.

Instructions (iv)
Follow these instructions:
- give each resource with a quote copied word for word from the page, which states that the service
  requires this resource.
- use no knowledge outside the page.
- exclude optional features and recommendations.
- exclude the parts that the provider creates automatically with the service and that are configured
  only through the attributes of the service; keep the resources that must be created or referenced
  for the service, including a resource that the provider supplies by default when none is specified.
- describe the properties to write in the template and the settings that the provider applies to the resource
  (e.g., its default settings) only with these properties, as property=value separated by ";":
{vocabulary}
- write {{region}} for the region of the deployment, never a region taken from an example of the page.
- if the page states no dependent resource, answer RESOURCE: none.

Output format (v)
RESOURCE: <name>
QUOTE: <passage copied word for word>
ATTACHED TO: <the service, or the name of another resource that requires it>
PROPERTIES: <properties written in the template, or none>
SETTINGS: <settings applied by the provider, per the page, or none>
(repeat these five lines for each resource)"""

URL_PROMPT = """Role (i)
You are an expert in cloud infrastructure and provider documentation.

Context (ii)
Considering the following service, for which no official documentation page is known:
Service: {service}
Provider: {provider}
Official domains of the provider: {domains}

Task (iii)
Propose the official documentation pages that describe the resources this service requires from its provider.

Instructions (iv)
Follow these instructions:
- propose at most {n} URLs, only on the official domains given above.
- propose only pages that exist; propose none if you do not know any.

Output format (v)
URL: <url>
(one line per URL, or URL: none)"""

REPAIR_PROMPT = """Role (i)
You are an expert in cloud infrastructure and provider documentation.

Context (ii)
Considering the following dependent resource of a service, and the official documentation page that requires it:
Service: {service}
Provider: {provider}
Resource: {resource}
Quote: {quote}
Current properties: {properties}
Settings applied by the provider: {settings}
Requirement it does not satisfy: {requirement}
Page: {page}

Task (iii)
Configure this resource again so that it satisfies the requirement.

Instructions (iv)
Follow these instructions:
- give the properties to write in the template, which override the settings of the provider.
- use only configurations that the page allows, and no knowledge outside the page.
- use only these properties, as property=value separated by ";":
{vocabulary}
- write {{region}} for the region of the deployment.

Output format (v)
PROPERTIES: <properties written in the template>"""
