"""
Candidate offers of each node of a provider-independent template, recomputed from the catalogs of our previous
work (data/catalogs/catalogs.json, 209 offers) with the deterministic matching of its discovery step
(requirement extraction, candidate filtering, candidate set construction): an offer is a candidate of a node when
it is of the node's service type and satisfies every requirement of the node, each treated as an independent
criterion. Cross-node constraints (global constraints) are not checked here: they belong to the selection.

Requirements extracted (when the template states them)
  node types     those of our previous work that represent cloud services (TYPE_TO_CATALOG); the other nodes are
                 software components hosted on them
  compute        num_cpus <= cpu, mem_size <= memory_gb, os.type in os_type, os.distribution in os_distribution,
                 os.architecture = architecture
  database       engine (from the DBMS name or properties), component_version in supported_versions,
                 num_cpus <= cpu, mem_size <= memory_gb (host capability of the DBMS; instance class of the offer)
  block storage  size <= capacity_gb
  load balancer  algorithm layer4 / layer7 = lb_type
  network        ip_version 4 / 6 among the ip_version of the offer
  DBMS           a DBMS hosted on a virtual machine is installed software, not a cloud service
  availability   every candidate has a published availability, field "sla": the SLA of the configuration whose
                 price is listed; a managed database is listed in each configuration covered by an SLA, single
                 zone and high availability (x2), the same rule for the three providers (priced_configurations);
                 the fields sla_single_zone and sla_multi_zone keep the values of the catalog, for reference only
  policies       Placement (region codes, or logical locations resolved by the region taxonomy),
                 Availability and Cost targeting only this node (a policy on several nodes is a global constraint)
Latency policies are not matched: the catalogs publish no latency.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from ..common.tosca import iter_policies, node_templates, policy_kind, policy_targets

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# node types of our previous work that represent cloud services, and their catalog (the other nodes, e.g. a web
# application, a Database hosted by a DBMS, or a container, are software components hosted on these services)
TYPE_TO_CATALOG = {
    "compute": "compute", "computewithnetwork": "compute", "computewithblocnetwork": "compute",
    "computewithblocstorage": "compute",
    "dbms": "database",
    "network": "network", "networkforcontainer": "network",
    "blockstorage": "block_storage",
    "objectstorage": "object_storage", "storageforcontainer": "object_storage",
    "loadbalancer": "loadbalancer",
}
LOCATION_ALIASES = {"us": "usa", "united states": "usa", "america": "usa", "eu": "europe", "united kingdom": "uk"}
ENGINES = [("postgres", "postgresql"), ("mysql", "mysql"), ("maria", "mariadb"), ("mongo", "mongodb"),
           ("sqlserver", "sqlserver"), ("sql server", "sqlserver"), ("oracle", "oracle"), ("redis", "redis")]


HA_PRICE_FACTOR = 2.0
# Price of a high-availability configuration of a managed database (a standby replica of the same size):
#   Google Cloud  "An HA-configured instance costs twice as much as a standalone instance."
#   Amazon RDS    "Multi-AZ deployments ... can cost approximately twice as much due to the additional standby instance"
#   Azure         "A standby replica is deployed in the same VM configuration - including vCores, storage, network
#                  settings - as the primary server."; zone redundancy "is not supported for ... Burstable compute tier"
HA_CONFIG = {
    "gcp": ("-ha", "high availability (regional)", "high availability"),
    "aws": ("-multiaz", "Multi-AZ DB instance (standby in a second Availability Zone)", "RDS Multi-AZ"),
    "azure": ("-zrha", "zone-redundant high availability (standby in a second zone)", "zone-redundant HA"),
}


def _config(o: dict, suffix: str, deployment: str, label: str, sla: float, factor: float) -> dict:
    x = dict(o)
    if suffix:
        x["offer_id"] = f"{o['offer_id']}{suffix}"
        x["offer_name"] = f"{o['offer_name']} ({label})"
    for k in ("price_hour", "price_month"):
        if _num(o.get(k)) is not None:
            x[k] = round(_num(o[k]) * factor, 4 if k == "price_hour" else 2)
    x["sla"] = sla
    x["deployment"] = deployment
    if factor != 1:
        x["price_reliability"] = f"{o.get('price_reliability')}, x{factor:g} for high availability"
    return x


def priced_configurations(o: dict) -> list[dict]:
    """The configurations of a managed-database offer that are covered by an SLA, each with its price and the SLA
    of that configuration (field "sla"); the same rule for the three providers:
      single zone        price of the catalog, single-zone SLA, when the provider publishes one (AWS, Azure);
                         Cloud SQL publishes none ("Shared-core instances and single-zone instances are not covered
                         by the SLA")
      high availability  price x2 (a standby replica of the same size), SLA of that configuration (multi-zone SLA of
                         the catalog), when the provider offers it for this instance: Cloud SQL with a dedicated CPU,
                         every RDS instance (Multi-AZ), Azure General Purpose (zone-redundant HA; not Burstable)
    An offer without any configuration covered by an SLA (shared-core Cloud SQL) is left out."""
    prov = o.get("provider")
    out = []
    single = _num(o.get("sla_single_zone"))
    if single is not None:
        out.append(_config(o, "", "single zone", "", single, 1.0))
    multi = _num(o.get("sla_multi_zone"))
    ha_allowed = {"gcp": (_num(o.get("cpu")) or 0) >= 1,
                  "aws": True,
                  "azure": "burstable" not in str(o.get("offer_name", "")).lower()}.get(prov, False)
    if multi is not None and ha_allowed and prov in HA_CONFIG:
        suffix, deployment, label = HA_CONFIG[prov]
        out.append(_config(o, suffix, deployment, label, multi, HA_PRICE_FACTOR))
    return out


def load_catalogs(path: Optional[str] = None) -> dict[str, list[dict]]:
    with open(path or os.path.join(DATA, "catalogs", "catalogs.json"), encoding="utf-8") as f:
        raw = json.load(f)["catalogs"]
    out = {}
    for k, offers in raw.items():
        out[k] = [x for o in offers for x in priced_configurations(o)] if k == "database" else offers
    return out


def load_c3_taxonomy(path: Optional[str] = None) -> dict[str, list[str]]:
    """logical location -> region codes of every provider."""
    with open(path or os.path.join(DATA, "catalogs", "region_taxonomy_c3.json"), encoding="utf-8") as f:
        rows = json.load(f)["taxonomy"]
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(str(r["logical_region"]).lower(), []).append(str(r["region_code"]))
    return out


def catalog_of(node_type: str) -> Optional[str]:
    t = re.sub(r"[^a-z]", "", (node_type or "").split(".")[-1].lower())
    return TYPE_TO_CATALOG.get(t)


def is_service(node: dict) -> bool:
    """A node of a service type is a cloud service, except a DBMS hosted on a virtual machine (installed software)."""
    cat = catalog_of(node.get("type"))
    if cat is None:
        return False
    if cat == "database" and any("host" in r for r in node.get("requirements") or [] if isinstance(r, dict)):
        return False
    return True


def _size_gb(v) -> Optional[float]:
    if v is None:
        return None
    m = re.match(r"\s*([\d.]+)\s*([kmgt]i?b?)?", str(v).lower())
    if not m:
        return None
    x = float(m.group(1))
    unit = (m.group(2) or "gb")[0]
    return x * {"k": 1e-6, "m": 1e-3, "g": 1, "t": 1000}[unit]


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _set(v) -> set[str]:
    return {x.strip().lower() for x in str(v or "").replace(",", ";").split(";") if x.strip() and x.strip() not in ("—", "n/a")}


def _caps(node: dict, cap: str) -> dict:
    return ((node.get("capabilities") or {}).get(cap) or {}).get("properties") or {}


def engine_of(name: str, node: dict) -> Optional[str]:
    text = " ".join([name, json.dumps(node.get("properties") or {}), json.dumps(_caps(node, "host"))]).lower()
    for k, e in ENGINES:
        if k in text:
            return e
    return None


def _regions(locations, taxonomy) -> Optional[set[str]]:
    regs: set[str] = set()
    for loc in locations if isinstance(locations, list) else [locations]:
        r = loc.get("region") if isinstance(loc, dict) else loc
        if not r:
            continue
        key = str(r).strip().lower()
        key = LOCATION_ALIASES.get(key, key)
        regs |= set(taxonomy.get(key, [])) or {str(r)}
    return regs or None


def node_requirements(tosca: dict, name: str, node: dict, all_nodes: set[str], taxonomy) -> dict:
    req: dict = {"catalog": catalog_of(node.get("type"))}
    host, os_ = _caps(node, "host"), _caps(node, "os")
    if req["catalog"] == "compute":
        req["cpu"] = _num(host.get("num_cpus"))
        req["mem_gb"] = _size_gb(host.get("mem_size"))
        req["os_type"] = (os_.get("type") or "").lower() or None
        req["os_distribution"] = (os_.get("distribution") or "").lower() or None
        req["architecture"] = (os_.get("architecture") or "").lower() or None
    elif req["catalog"] == "database":
        req["cpu"] = _num(host.get("num_cpus"))
        req["mem_gb"] = _size_gb(host.get("mem_size"))
        req["engine"] = engine_of(name, node)
        v = (node.get("properties") or {}).get("component_version")
        req["version"] = str(v).split(".")[0] if v not in (None, "") else None
    elif req["catalog"] == "block_storage":
        req["size_gb"] = _size_gb((node.get("properties") or {}).get("size"))
    elif req["catalog"] == "loadbalancer":
        alg = str((node.get("properties") or {}).get("algorithm") or "").lower()
        req["lb_type"] = alg if alg in ("layer4", "layer7") else None
    elif req["catalog"] == "network":
        ipv = (node.get("properties") or {}).get("ip_version")
        req["ip_version"] = f"ipv{int(ipv)}" if _num(ipv) in (4.0, 6.0) else None
    for pname, body in iter_policies(tosca):
        targets = policy_targets(tosca, body, all_nodes) if body.get("targets") else list(all_nodes)
        if name not in targets:
            continue
        kind, props = policy_kind(body.get("type")), body.get("properties") or {}
        if kind == "placement":
            regs = _regions(props.get("locations") or props.get("location"), taxonomy)
            if regs:
                req["regions"] = (req.get("regions") or regs) & regs
        elif len(targets) == 1 and kind == "availability":
            req["min_availability"] = _num(props.get("availability") or props.get("min_availability"))
        elif len(targets) == 1 and kind == "cost":
            req["max_price"] = _num(props.get("max_cost") or props.get("max_monthly_price"))
    return req


def _sla(o: dict) -> Optional[float]:
    for k in ("sla", "sla_single_zone"):          # "sla": SLA of the priced configuration, when set
        v = _num(o.get(k))
        if v is not None:
            return v
    return None


def satisfies(o: dict, req: dict) -> bool:
    c = req["catalog"]
    if req.get("regions") and str(o.get("region")) not in req["regions"] and str(o.get("region")) != "global":
        return False
    if c in ("compute", "database"):
        if req.get("cpu") and (_num(o.get("cpu")) or 0) < req["cpu"]:
            return False
        if req.get("mem_gb") and (_num(o.get("memory_gb")) or 0) < req["mem_gb"] - 1e-9:
            return False
    if c == "compute":
        if req.get("os_type") and req["os_type"] not in _set(o.get("os_type")):
            return False
        if req.get("os_distribution") and req["os_distribution"] not in _set(o.get("os_distribution")):
            return False
        arch = req.get("architecture")
        if arch:
            a = {"amd64": "x86_64", "x86-64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(arch, arch)
            if a != str(o.get("architecture") or "").lower():
                return False
    elif c == "database":
        if req.get("engine") and req["engine"] != str(o.get("engine") or "").lower():
            return False
        if req.get("version") and req["version"] not in {v.split(".")[0] for v in _set(o.get("supported_versions"))}:
            return False
    elif c == "block_storage":
        if req.get("size_gb") and (_num(o.get("capacity_gb")) or 0) < req["size_gb"]:
            return False
    elif c == "loadbalancer":
        if req.get("lb_type") and req["lb_type"] != str(o.get("lb_type") or "").lower():
            return False
    elif c == "network":
        if req.get("ip_version") and req["ip_version"] not in _set(o.get("ip_version")):
            return False
    if req.get("min_availability") is not None:
        s = _sla(o)
        if s is None or s < req["min_availability"]:
            return False
    if req.get("max_price") is not None:
        p = _num(o.get("price_month"))
        if p is None or p > req["max_price"]:
            return False
    return True


def to_candidate(o: dict, catalog: str, req: dict) -> dict:
    out = {k: o.get(k) for k in ("offer_id", "offer_name", "provider", "region")}
    out["offer_type"] = "database" if catalog == "database" else catalog
    if catalog == "database":
        out["engine"] = o.get("engine")
    price = _num(o.get("price_month"))
    if price is None and _num(o.get("price_gb")) is not None:       # object storage: price per GB
        price = round(_num(o.get("price_gb")) * (req.get("size_gb") or 100), 2)
    out["price_month"] = price if price is not None else 0.0
    out["sla"] = _sla(o)                             # availability of the priced configuration
    out["sla_single_zone"] = _num(o.get("sla_single_zone"))   # catalog values, for reference only
    out["sla_multi_zone"] = _num(o.get("sla_multi_zone"))
    if o.get("deployment"):
        out["deployment"] = o["deployment"]
    return out


def discover(tosca: dict, catalogs: Optional[dict] = None, taxonomy: Optional[dict] = None) -> dict:
    """Discovery output (format of our previous work) and the unsatisfiable nodes."""
    catalogs = catalogs or load_catalogs()
    taxonomy = taxonomy or load_c3_taxonomy()
    nodes = node_templates(tosca)
    service_nodes = {n for n, b in nodes.items() if is_service(b)}
    out, unsat = [], []
    for name in nodes:
        if name not in service_nodes:
            continue
        req = node_requirements(tosca, name, nodes[name], service_nodes, taxonomy)
        cands = [to_candidate(o, req["catalog"], req) for o in catalogs.get(req["catalog"], []) if satisfies(o, req)]
        if not cands:
            unsat.append({"node_name": name, "requirements": {k: (sorted(v) if isinstance(v, set) else v)
                                                              for k, v in req.items()}})
            continue
        out.append({"node_name": name, "applied_filters": {"sla_field_used": "sla",
                                                           "requirements": {k: (sorted(v) if isinstance(v, set) else v)
                                                                            for k, v in req.items()}},
                    "candidates": cands})
    return {"currency": "USD", "nodes": out, "unsatisfiable": unsat}
