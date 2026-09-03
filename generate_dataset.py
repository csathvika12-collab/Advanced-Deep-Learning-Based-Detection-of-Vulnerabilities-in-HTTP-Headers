"""
Generate a 5,000-record HTTP header vulnerability training dataset.
Pure Python stdlib — no numpy or pandas required.
Each vulnerability type gets an explicit class label so the distribution
is balanced: ~20% Secure | ~20% Low Risk | ~35% Medium Risk | ~25% High Risk

Run:   python generate_dataset.py
Output: training_dataset.csv  (5,000 rows, 60 columns)
"""

import csv, random, re, os, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
random.seed(42)

CLASS_NAMES = {0: 'Secure', 1: 'Low Risk', 2: 'Medium Risk', 3: 'High Risk'}

# ── Sample pools ──────────────────────────────────────────────────────────────
SERVERS = [
    '', 'nginx', 'nginx/1.18.0', 'nginx/1.14.2 (Ubuntu)', 'nginx/1.24.0',
    'Apache', 'Apache/2.4.51 (Unix)', 'Apache/2.4.41 (Ubuntu)', 'Apache/2.2.34',
    'Microsoft-IIS/10.0', 'Microsoft-IIS/8.5', 'Microsoft-IIS/7.5',
    'cloudflare', 'Caddy', 'LiteSpeed', 'LiteSpeed/5.4',
    'openresty/1.19.9.1', 'gunicorn/20.1.0', 'uvicorn', 'Kestrel',
    'Tengine', 'Tengine/2.3.3', 'WEBrick/1.4.2',
]
POWERED_BY = [
    '', 'PHP/7.4.3', 'PHP/8.0.12', 'PHP/8.1.0', 'PHP/5.6.40', 'PHP/7.0.33',
    'ASP.NET', 'ASP.NET MVC 5.2', 'Express', 'Django/3.2', 'Django/4.0',
    'Ruby on Rails 6.1', 'Laravel', 'WordPress 6.1', 'Drupal 9',
    'Joomla! 3.9', 'Plone/5.2', 'Next.js',
]
DOMAINS = [
    'example.com','myblog.net','shopsite.org','govportal.gov',
    'university.edu','startup.io','enterprise.co','newssite.com',
    'bank.com','hospital.org','store.net','forum.co','portal.net',
    'api.example.com','admin.example.com','legacy.example.com',
    'cdn.example.com','mail.example.com','login.example.com',
    'dashboard.app.io','user.myservice.com','checkout.shop.com',
    'secure.finance.org','app.saas.com','cms.agency.net',
    'support.bigcorp.com','docs.devtool.io','status.service.net',
]
CSP_GOOD = [
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
    "default-src 'self' https:; script-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'",
    "default-src 'none'; script-src 'self'; connect-src 'self'; img-src 'self'; style-src 'self'",
    "default-src 'self'; upgrade-insecure-requests; block-all-mixed-content",
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self'; img-src 'self' https:; frame-ancestors 'none'",
]
CSP_INLINE = [
    "default-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
    "script-src 'self' 'unsafe-inline' https://cdn.example.com; style-src 'unsafe-inline'",
    "default-src * 'unsafe-inline' 'unsafe-eval'; img-src *",
    "script-src * 'unsafe-inline'; object-src 'self'",
]
CSP_EVAL = [
    "script-src 'self' 'unsafe-eval'; style-src 'self'",
    "default-src 'self' 'unsafe-eval' 'unsafe-inline'",
]
CSP_WILD = [
    "default-src *; script-src * 'unsafe-inline'",
    "default-src *; img-src *; connect-src *",
]

COLS = [
    'url','status','https','header_count',
    'has_strict_transport_security','hsts_max_age','hsts_include_subdomains',
    'hsts_preload','hsts_valid',
    'has_content_security_policy','csp_present','csp_unsafe_inline',
    'csp_unsafe_eval','csp_wildcard','csp_allow_data','csp_directive_count',
    'has_x_frame_options','xfo_deny','xfo_sameorigin','xfo_allowfrom',
    'has_x_content_type_options','xcto_nosniff',
    'has_x_xss_protection','xxss_enabled','xxss_block','xxss_report_uri',
    'has_referrer_policy','rp_no_referrer','rp_same_origin','rp_strict_origin',
    'rp_unsafe_url','rp_no_restriction',
    'has_permissions_policy','has_cache_control','cc_no_store','cc_no_cache',
    'cc_public_sensitive',
    'has_expect_ct',
    'has_cross_origin_embedder_policy','has_coep',
    'has_cross_origin_opener_policy','has_coop',
    'has_cross_origin_resource_policy','has_corp',
    'has_access_control_allow_origin','has_access_control_allow_credentials',
    'cors_wildcard','cors_cred_wildcard','cors_present',
    'cookie_present','cookie_secure','cookie_httponly','cookie_samesite',
    'info_leak_count','server_version_exposed','x_powered_by_present',
    'header_text',
    'vuln_class','vuln_label','is_vulnerable',
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build raw header text string from a feature dict
# ─────────────────────────────────────────────────────────────────────────────
def _htext(r):
    p = []
    if r.get('has_strict_transport_security'):
        age = r.get('hsts_max_age', 31536000)
        v   = f"max-age={age}"
        if r.get('hsts_include_subdomains'): v += '; includeSubDomains'
        if r.get('hsts_preload'):            v += '; preload'
        p.append(f"strict-transport-security: {v}")
    if r.get('csp_present'):
        if   r.get('csp_unsafe_inline') and r.get('csp_unsafe_eval'):
            p.append(f"content-security-policy: {random.choice(CSP_EVAL)}")
        elif r.get('csp_unsafe_inline'):
            p.append(f"content-security-policy: {random.choice(CSP_INLINE)}")
        elif r.get('csp_wildcard'):
            p.append(f"content-security-policy: {random.choice(CSP_WILD)}")
        else:
            p.append(f"content-security-policy: {random.choice(CSP_GOOD)}")
    if r.get('has_x_frame_options'):
        xv = ('DENY' if r.get('xfo_deny') else
              ('SAMEORIGIN' if r.get('xfo_sameorigin') else 'ALLOW-FROM https://partner.example.com'))
        p.append(f"x-frame-options: {xv}")
    if r.get('xcto_nosniff'): p.append("x-content-type-options: nosniff")
    if r.get('has_referrer_policy'):
        rp = ('unsafe-url' if r.get('rp_unsafe_url') else
              ('no-referrer' if r.get('rp_no_referrer') else
               'strict-origin-when-cross-origin'))
        p.append(f"referrer-policy: {rp}")
    if r.get('has_permissions_policy'):
        p.append("permissions-policy: camera=(), microphone=(), geolocation=()")
    if r.get('has_cache_control'):
        cc = ('no-store, no-cache' if r.get('cc_no_store') else
              ('public, max-age=86400' if r.get('cc_public_sensitive') else 'private, max-age=0'))
        p.append(f"cache-control: {cc}")
    sv = r.get('_srv', '')
    if sv: p.append(f"server: {sv}")
    if r.get('x_powered_by_present'):
        p.append(f"x-powered-by: {random.choice([x for x in POWERED_BY if x])}")
    if r.get('cors_wildcard'):        p.append("access-control-allow-origin: *")
    if r.get('cors_cred_wildcard'):   p.append("access-control-allow-credentials: true")
    if r.get('cookie_present'):
        ck = "session=abc123"
        if r.get('cookie_secure'):   ck += "; Secure"
        if r.get('cookie_httponly'): ck += "; HttpOnly"
        if r.get('cookie_samesite'): ck += "; SameSite=Strict"
        p.append(f"set-cookie: {ck}")
    if r.get('has_coep'): p.append("cross-origin-embedder-policy: require-corp")
    if r.get('has_coop'): p.append("cross-origin-opener-policy: same-origin")
    if r.get('has_corp'): p.append("cross-origin-resource-policy: same-site")
    p += ["content-type: text/html; charset=utf-8", "date: Mon, 01 Jan 2024 00:00:00 GMT"]
    return " | ".join(p)

# ─────────────────────────────────────────────────────────────────────────────
# Full-feature row builders — class is set EXPLICITLY so distribution is exact
# ─────────────────────────────────────────────────────────────────────────────

def _row(overrides, forced_class):
    """Build a complete feature row with safe defaults then apply overrides."""
    r = dict(
        url=f"https://{random.choice(DOMAINS)}/page",
        status=random.choice([200,200,200,200,301,302]),
        https=1, header_count=random.randint(10,22),
        # HSTS
        has_strict_transport_security=1, hsts_max_age=random.choice([31536000,63072000]),
        hsts_include_subdomains=1, hsts_preload=random.choice([0,1]), hsts_valid=1,
        # CSP
        has_content_security_policy=1, csp_present=1, csp_unsafe_inline=0,
        csp_unsafe_eval=0, csp_wildcard=0, csp_allow_data=0,
        csp_directive_count=random.randint(4,9),
        # XFO
        has_x_frame_options=1, xfo_deny=1, xfo_sameorigin=0, xfo_allowfrom=0,
        # XCTO
        has_x_content_type_options=1, xcto_nosniff=1,
        # XSS
        has_x_xss_protection=1, xxss_enabled=1, xxss_block=1, xxss_report_uri=0,
        # Referrer
        has_referrer_policy=1, rp_no_referrer=0, rp_same_origin=1,
        rp_strict_origin=1, rp_unsafe_url=0, rp_no_restriction=0,
        # Misc
        has_permissions_policy=1, has_cache_control=1,
        cc_no_store=1, cc_no_cache=1, cc_public_sensitive=0,
        has_expect_ct=random.choice([0,1]),
        # Modern isolation
        has_cross_origin_embedder_policy=1, has_coep=1,
        has_cross_origin_opener_policy=1,   has_coop=1,
        has_cross_origin_resource_policy=1, has_corp=1,
        # CORS
        has_access_control_allow_origin=0, has_access_control_allow_credentials=0,
        cors_wildcard=0, cors_cred_wildcard=0, cors_present=0,
        # Cookies
        cookie_present=random.choice([0,1]),
        cookie_secure=1, cookie_httponly=1, cookie_samesite=1,
        # Info leak
        info_leak_count=0, server_version_exposed=0, x_powered_by_present=0,
        _srv='',
    )
    r.update(overrides)
    r['header_text'] = _htext(r)
    r['vuln_class']  = forced_class
    r['vuln_label']  = CLASS_NAMES[forced_class]
    r['is_vulnerable'] = 1 if forced_class > 0 else 0
    r.pop('_srv', None)
    return r

def _vsrv():
    return random.choice([s for s in SERVERS if re.search(r'[0-9]', s)])

# ══════════════════════════════════════════════════════════════════════════════
# CLASS 0 — SECURE  (1 profile, 1000 records)
# ══════════════════════════════════════════════════════════════════════════════

def secure():
    """Fully hardened site — all critical headers present and correctly configured."""
    return _row({
        'hsts_preload': random.choice([0,1]),
        'hsts_include_subdomains': 1,
        'hsts_max_age': random.choice([31536000, 63072000]),
        'csp_directive_count': random.randint(5,9),
        'cookie_present': random.choice([0,1]),
        'header_count': random.randint(14,24),
    }, forced_class=0)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1 — LOW RISK  (10 profiles, 1000 records)
# ══════════════════════════════════════════════════════════════════════════════

def low_hsts_short():
    """HSTS present but max-age under 1 year."""
    return _row({
        'hsts_max_age': random.choice([604800,2592000,7776000,15552000]),
        'hsts_valid': 0, 'hsts_preload': 0,
    }, forced_class=1)

def low_no_referrer_policy():
    """Referrer-Policy header absent — minor info-leak risk."""
    return _row({
        'has_referrer_policy': 0, 'rp_same_origin': 0, 'rp_strict_origin': 0,
    }, forced_class=1)

def low_no_permissions_policy():
    """Permissions-Policy absent — browser features not restricted."""
    return _row({'has_permissions_policy': 0}, forced_class=1)

def low_no_coep_coop():
    """Missing cross-origin isolation headers (COEP/COOP/CORP)."""
    return _row({
        'has_coep':0,'has_coop':0,'has_corp':0,
        'has_cross_origin_embedder_policy':0,
        'has_cross_origin_opener_policy':0,
        'has_cross_origin_resource_policy':0,
    }, forced_class=1)

def low_xfo_allowfrom():
    """X-Frame-Options uses deprecated ALLOW-FROM directive."""
    return _row({'xfo_deny':0,'xfo_sameorigin':0,'xfo_allowfrom':1}, forced_class=1)

def low_xxss_disabled():
    """X-XSS-Protection absent or disabled."""
    return _row({'xxss_enabled':0,'xxss_block':0,'has_x_xss_protection':0}, forced_class=1)

def low_no_cache_control():
    """Cache-Control header absent."""
    return _row({'has_cache_control':0,'cc_no_store':0,'cc_no_cache':0}, forced_class=1)

def low_hsts_no_subdomains():
    """HSTS present but missing includeSubDomains and preload."""
    return _row({'hsts_include_subdomains':0,'hsts_preload':0}, forced_class=1)

def low_csp_data_uri():
    """CSP allows data: URIs — minor weakness."""
    return _row({'csp_allow_data':1,'csp_directive_count':random.randint(3,6)}, forced_class=1)

def low_minor_leak():
    """X-Powered-By header reveals backend technology."""
    return _row({'x_powered_by_present':1,'info_leak_count':1}, forced_class=1)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2 — MEDIUM RISK  (15 profiles, 1750 records)
# ══════════════════════════════════════════════════════════════════════════════

def med_no_hsts():
    """HSTS completely absent — browser won't enforce HTTPS."""
    return _row({
        'has_strict_transport_security':0,'hsts_max_age':0,
        'hsts_include_subdomains':0,'hsts_preload':0,'hsts_valid':0,
    }, forced_class=2)

def med_no_xfo():
    """X-Frame-Options absent — clickjacking risk."""
    return _row({'has_x_frame_options':0,'xfo_deny':0,'xfo_sameorigin':0}, forced_class=2)

def med_no_xcto():
    """X-Content-Type-Options: nosniff absent — MIME-sniffing risk."""
    return _row({'xcto_nosniff':0,'has_x_content_type_options':0}, forced_class=2)

def med_unsafe_referrer():
    """Referrer-Policy set to unsafe-url — leaks full URL to third parties."""
    return _row({
        'rp_unsafe_url':1,'rp_same_origin':0,'rp_strict_origin':0,'rp_no_restriction':0,
    }, forced_class=2)

def med_cors_wildcard():
    """CORS: Access-Control-Allow-Origin: * — any site can read API responses."""
    return _row({
        'cors_wildcard':1,'cors_present':1,'has_access_control_allow_origin':1,
    }, forced_class=2)

def med_cookie_no_secure():
    """Session cookie missing Secure flag — sent over HTTP too."""
    return _row({'cookie_present':1,'cookie_secure':0}, forced_class=2)

def med_cookie_no_httponly():
    """Session cookie missing HttpOnly — accessible via JavaScript (XSS theft)."""
    return _row({'cookie_present':1,'cookie_httponly':0}, forced_class=2)

def med_cookie_no_samesite():
    """Session cookie missing SameSite attribute — CSRF risk."""
    return _row({'cookie_present':1,'cookie_samesite':0}, forced_class=2)

def med_server_version():
    """Server header reveals exact version string — aids CVE targeting."""
    return _row({
        '_srv':_vsrv(),'server_version_exposed':1,
        'info_leak_count':random.randint(1,3),
    }, forced_class=2)

def med_cache_public():
    """Cache-Control: public on sensitive page — proxies may cache private data."""
    return _row({'cc_no_store':0,'cc_no_cache':0,'cc_public_sensitive':1}, forced_class=2)

def med_api_cors():
    """REST API using broad CORS — common misconfiguration."""
    return _row({
        'url':f"https://api.{random.choice(DOMAINS)}/v1/resource",
        'has_content_security_policy':0,'csp_present':0,
        'has_x_frame_options':0,'xfo_deny':0,
        'cors_present':1,'has_access_control_allow_origin':1,
        'cors_wildcard':random.choice([0,1]),
    }, forced_class=2)

def med_legacy():
    """Legacy site — missing several modern headers but HSTS/CSP/XFO intact."""
    return _row({
        'has_permissions_policy':0,
        'has_coep':0,'has_coop':0,'has_corp':0,
        'has_cross_origin_embedder_policy':0,
        'has_cross_origin_opener_policy':0,
        'has_cross_origin_resource_policy':0,
        'has_expect_ct':0,
        'x_powered_by_present':1,'info_leak_count':random.randint(2,3),
        '_srv':_vsrv(),'server_version_exposed':1,
    }, forced_class=2)

def med_missing_three():
    """Missing 3 of the 5 important security headers."""
    combos = [
        {'has_referrer_policy':0,'has_permissions_policy':0,'xcto_nosniff':0,'has_x_content_type_options':0},
        {'has_referrer_policy':0,'has_permissions_policy':0,'has_cache_control':0,'cc_no_store':0},
        {'xcto_nosniff':0,'has_x_content_type_options':0,'has_permissions_policy':0,'cc_public_sensitive':1,'cc_no_store':0},
        {'has_referrer_policy':0,'cc_public_sensitive':1,'cc_no_store':0,'x_powered_by_present':1,'info_leak_count':1},
    ]
    return _row(random.choice(combos), forced_class=2)

def med_xpowered_versioned():
    """Both X-Powered-By and versioned Server header present."""
    return _row({
        '_srv':_vsrv(),'server_version_exposed':1,
        'x_powered_by_present':1,'info_leak_count':random.randint(2,4),
    }, forced_class=2)

def med_mixed():
    """2–4 minor issues combined — realistic partially-hardened site."""
    overrides = {}
    tweaks = random.sample([
        {'has_referrer_policy':0,'rp_same_origin':0},
        {'has_permissions_policy':0},
        {'cc_no_store':0,'cc_public_sensitive':1},
        {'x_powered_by_present':1,'info_leak_count':1},
        {'hsts_preload':0,'hsts_include_subdomains':0},
        {'xxss_block':0},
        {'has_coep':0,'has_cross_origin_embedder_policy':0},
        {'csp_allow_data':1},
        {'has_cache_control':0,'cc_no_store':0,'cc_no_cache':0},
        {'cookie_present':1,'cookie_samesite':0},
    ], k=random.randint(2,4))
    for t in tweaks: overrides.update(t)
    return _row(overrides, forced_class=2)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS 3 — HIGH RISK  (13 profiles, 1250 records)
# ══════════════════════════════════════════════════════════════════════════════

def high_http_only():
    """Site served over plain HTTP — all traffic unencrypted."""
    ov = {
        'url':f"http://{random.choice(DOMAINS)}/page",
        'https':0,'has_strict_transport_security':0,
        'hsts_max_age':0,'hsts_valid':0,'hsts_include_subdomains':0,'hsts_preload':0,
    }
    if random.random() < 0.6:
        ov.update({'has_content_security_policy':0,'csp_present':0})
    if random.random() < 0.5:
        ov.update({'has_x_frame_options':0,'xfo_deny':0})
    return _row(ov, forced_class=3)

def high_no_csp():
    """No Content-Security-Policy — XSS and injection fully unmitigated."""
    return _row({
        'has_content_security_policy':0,'csp_present':0,'csp_directive_count':0,
    }, forced_class=3)

def high_csp_unsafe_inline():
    """CSP contains 'unsafe-inline' — XSS protection negated."""
    return _row({
        'csp_unsafe_inline':1,'csp_directive_count':random.randint(2,5),
    }, forced_class=3)

def high_csp_unsafe_eval():
    """CSP contains 'unsafe-eval' — eval() and similar functions allowed."""
    return _row({
        'csp_unsafe_eval':1,'csp_directive_count':random.randint(2,5),
    }, forced_class=3)

def high_csp_both_unsafe():
    """CSP has both unsafe-inline and unsafe-eval — CSP is essentially useless."""
    return _row({
        'csp_unsafe_inline':1,'csp_unsafe_eval':1,
        'csp_directive_count':random.randint(1,3),
    }, forced_class=3)

def high_csp_wildcard():
    """CSP uses wildcard (*) — any script source allowed."""
    return _row({
        'csp_wildcard':1,'csp_unsafe_inline':random.choice([0,1]),
        'csp_directive_count':random.randint(1,3),
    }, forced_class=3)

def high_cors_cred_wildcard():
    """CRITICAL: CORS wildcard + credentials — any origin reads authenticated responses."""
    return _row({
        'cors_wildcard':1,'cors_present':1,'cors_cred_wildcard':1,
        'has_access_control_allow_origin':1,'has_access_control_allow_credentials':1,
    }, forced_class=3)

def high_insecure_cookies():
    """Session cookies missing Secure, HttpOnly, and SameSite — full cookie exposure."""
    return _row({
        'cookie_present':1,'cookie_secure':0,'cookie_httponly':0,'cookie_samesite':0,
    }, forced_class=3)

def high_multi_info_leak():
    """5+ technology-disclosure headers — full stack fingerprint exposed."""
    return _row({
        '_srv':_vsrv(),'server_version_exposed':1,
        'x_powered_by_present':1,'info_leak_count':random.randint(4,7),
    }, forced_class=3)

def high_no_csp_no_xfo():
    """No CSP + No X-Frame-Options — XSS + clickjacking simultaneously exposed."""
    return _row({
        'has_content_security_policy':0,'csp_present':0,
        'has_x_frame_options':0,'xfo_deny':0,
    }, forced_class=3)

def high_no_csp_insecure_cookies():
    """No CSP combined with insecure cookies — session hijack via XSS."""
    return _row({
        'has_content_security_policy':0,'csp_present':0,
        'cookie_present':1,'cookie_secure':0,'cookie_httponly':0,
    }, forced_class=3)

def high_http_no_csp_no_hsts():
    """Plain HTTP + no HSTS + no CSP + version disclosed — severely misconfigured."""
    return _row({
        'url':f"http://{random.choice(DOMAINS)}/page",
        'https':0,'has_strict_transport_security':0,
        'hsts_max_age':0,'hsts_valid':0,'hsts_include_subdomains':0,
        'has_content_security_policy':0,'csp_present':0,
        '_srv':_vsrv(),'server_version_exposed':1,
        'info_leak_count':random.randint(2,5),
    }, forced_class=3)

def high_critically_broken():
    """Everything wrong — comprehensive security failure."""
    return _row({
        'has_strict_transport_security':0,'hsts_max_age':0,'hsts_valid':0,
        'hsts_include_subdomains':0,'hsts_preload':0,
        'has_content_security_policy':0,'csp_present':0,
        'has_x_frame_options':0,'xfo_deny':0,
        'xcto_nosniff':0,'has_x_content_type_options':0,
        'cors_wildcard':1,'cors_cred_wildcard':1,'cors_present':1,
        'has_access_control_allow_origin':1,'has_access_control_allow_credentials':1,
        'cookie_present':1,'cookie_secure':0,'cookie_httponly':0,'cookie_samesite':0,
        '_srv':_vsrv(),'server_version_exposed':1,
        'x_powered_by_present':1,'info_leak_count':random.randint(4,7),
        'has_permissions_policy':0,'has_coep':0,'has_coop':0,
        'has_cross_origin_embedder_policy':0,'has_cross_origin_opener_policy':0,
        'rp_unsafe_url':1,'rp_same_origin':0,'rp_strict_origin':0,
    }, forced_class=3)

# ─────────────────────────────────────────────────────────────────────────────
# Profile table  (generator, count)  → total must sum to 5000
# ─────────────────────────────────────────────────────────────────────────────
PROFILES = [
    # ── Class 0 — Secure ─── 1000 records (20%)
    (secure,                     1000),

    # ── Class 1 — Low Risk ── 1000 records (20%)
    (low_hsts_short,              110),
    (low_no_referrer_policy,      110),
    (low_no_permissions_policy,   110),
    (low_no_coep_coop,            110),
    (low_xfo_allowfrom,            90),
    (low_xxss_disabled,            90),
    (low_no_cache_control,         90),
    (low_hsts_no_subdomains,       90),
    (low_csp_data_uri,             100),
    (low_minor_leak,              100),

    # ── Class 2 — Medium Risk ── 1750 records (35%)
    (med_no_hsts,                 140),
    (med_no_xfo,                  130),
    (med_no_xcto,                 120),
    (med_unsafe_referrer,         100),
    (med_cors_wildcard,           120),
    (med_cookie_no_secure,        110),
    (med_cookie_no_httponly,      110),
    (med_cookie_no_samesite,      100),
    (med_server_version,          110),
    (med_cache_public,             90),
    (med_api_cors,                100),
    (med_legacy,                  100),
    (med_missing_three,           100),
    (med_xpowered_versioned,       90),
    (med_mixed,                   230),

    # ── Class 3 — High Risk ── 1250 records (25%)
    (high_http_only,              110),
    (high_no_csp,                 130),
    (high_csp_unsafe_inline,      110),
    (high_csp_unsafe_eval,         90),
    (high_csp_both_unsafe,         90),
    (high_csp_wildcard,            80),
    (high_cors_cred_wildcard,      90),
    (high_insecure_cookies,       110),
    (high_multi_info_leak,         80),
    (high_no_csp_no_xfo,          100),
    (high_no_csp_insecure_cookies, 90),
    (high_http_no_csp_no_hsts,    100),
    (high_critically_broken,       70),
]

# ─────────────────────────────────────────────────────────────────────────────
# Generate rows & Write CSV
# ─────────────────────────────────────────────────────────────────────────────
def main(output_path='training_dataset.csv'):
    print("Generating 5,000-record dataset…")
    rows = []
    for gen_fn, count in PROFILES:
        for _ in range(count):
            rows.append(gen_fn())

    assert len(rows) == 5000, f"Row count mismatch: {len(rows)}"
    random.shuffle(rows)

    OUT = output_path
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    class_counts = {0:0,1:0,2:0,3:0}
    vt = {
        'No HTTPS':0,'No HSTS':0,'HSTS too short (<1yr)':0,'No CSP':0,
        "CSP 'unsafe-inline'":0,"CSP 'unsafe-eval'":0,'CSP wildcard (*)':0,
        'No X-Frame-Options':0,'No X-Content-Type-Options':0,'No Referrer-Policy':0,
        'Referrer unsafe-url':0,'No Permissions-Policy':0,
        'CORS wildcard':0,'CORS wildcard + credentials':0,
        'Cookie: no Secure flag':0,'Cookie: no HttpOnly flag':0,
        'Cookie: no SameSite attr':0,'Server version exposed':0,
        'X-Powered-By present':0,'Info-leak ≥3 headers':0,
        'No Cache-Control':0,'Cache-Control: public':0,'No COEP/COOP':0,
    }
    for r in rows:
        cls = r['vuln_class']
        class_counts[cls] += 1
        if not r.get('https',1):               vt['No HTTPS'] += 1
        if not r.get('has_strict_transport_security',0): vt['No HSTS'] += 1
        if r.get('has_strict_transport_security') and r.get('hsts_max_age',0)<31536000:
            vt['HSTS too short (<1yr)'] += 1
        if not r.get('csp_present',0):         vt['No CSP'] += 1
        if r.get('csp_unsafe_inline'):         vt["CSP 'unsafe-inline'"] += 1
        if r.get('csp_unsafe_eval'):           vt["CSP 'unsafe-eval'"] += 1
        if r.get('csp_wildcard'):              vt['CSP wildcard (*)'] += 1
        if not r.get('has_x_frame_options',0): vt['No X-Frame-Options'] += 1
        if not r.get('xcto_nosniff',0):        vt['No X-Content-Type-Options'] += 1
        if not r.get('has_referrer_policy',0): vt['No Referrer-Policy'] += 1
        if r.get('rp_unsafe_url'):             vt['Referrer unsafe-url'] += 1
        if not r.get('has_permissions_policy',0): vt['No Permissions-Policy'] += 1
        if r.get('cors_wildcard'):             vt['CORS wildcard'] += 1
        if r.get('cors_cred_wildcard'):        vt['CORS wildcard + credentials'] += 1
        if r.get('cookie_present') and not r.get('cookie_secure'):   vt['Cookie: no Secure flag'] += 1
        if r.get('cookie_present') and not r.get('cookie_httponly'): vt['Cookie: no HttpOnly flag'] += 1
        if r.get('cookie_present') and not r.get('cookie_samesite'): vt['Cookie: no SameSite attr'] += 1
        if r.get('server_version_exposed'):    vt['Server version exposed'] += 1
        if r.get('x_powered_by_present'):      vt['X-Powered-By present'] += 1
        if r.get('info_leak_count',0)>=3:      vt['Info-leak ≥3 headers'] += 1
        if not r.get('has_cache_control',0):   vt['No Cache-Control'] += 1
        if r.get('cc_public_sensitive'):       vt['Cache-Control: public'] += 1
        if not r.get('has_coep',0) and not r.get('has_coop',0): vt['No COEP/COOP'] += 1

    total = len(rows)
    print(f"\n{'='*60}")
    print(f"  DATASET: {OUT}")
    print(f"{'='*60}")
    print(f"  Total rows  : {total:,}")
    print(f"  Columns     : {len(COLS)}")
    print(f"  File size   : {os.path.getsize(OUT)//1024:,} KB\n")

    print("  Class Distribution:")
    for cls in range(4):
        cnt = class_counts[cls]
        pct = cnt/total*100
        bar = '█' * int(pct/2)
        print(f"    {CLASS_NAMES[cls]:12s} (class {cls}): {cnt:5,}  {pct:5.1f}%  {bar}")

    print(f"\n  Vulnerability-Type Breakdown (23 types):")
    for name, cnt in vt.items():
        pct = cnt/total*100
        mark = '✓' if cnt > 0 else '✗'
        print(f"  {mark}  {name:<38s}: {cnt:5,}  ({pct:4.1f}%)")

    safe = class_counts[0]
    vuln = total - safe
    print(f"\n  Secure rows     : {safe:,}  ({safe/total*100:.1f}%)")
    print(f"  Vulnerable rows : {vuln:,}  ({vuln/total*100:.1f}%)")
    print(f"\n✅ Saved → {OUT}")
    return OUT

if __name__ == '__main__':
    main()
