"""Run the shipped login script against a deterministic provider, without credentials."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def run_login(query, scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser contract check")
    page = Path("backend/account_login.html").read_text(encoding="utf-8")
    script = re.search(r'<script nonce="__NONCE__">(.*?)</script>', page, re.S).group(1)
    config = {
        "provider": "https://test.neonauth.neon.tech/neondb/auth",
        "origin": "https://ragbench.example",
        "dashboard": "https://ragbench.streamlit.app",
        "flow": "f" * 32,
    }
    harness = r"""
const vm = require('node:vm');
const elements = new Map(), calls = [], destinations = [], clean = [];
const element = id => {if(!elements.has(id))elements.set(id,{hidden:false,value:'',textContent:'',disabled:false});return elements.get(id)};
const context = {
  URL,URLSearchParams,AbortSignal,
  document:{getElementById:element,querySelector:element,querySelectorAll:()=>[]},
  location:{search:QUERY,href:'https://ragbench.example/auth/login'+QUERY,
    assign:url=>destinations.push(url),replace:url=>destinations.push(url)},
  history:{replaceState:(_,__,url)=>clean.push(url)},
  fetch:async(url,init)=>{
    calls.push({url,method:init?.method||'GET'});
    if(String(url).includes('/get-session'))return {ok:SCENARIO!=='failed',json:async()=>SCENARIO==='callback'?{user:{emailVerified:true}}:null};
    if(String(url).endsWith('/token'))return {ok:true,json:async()=>({token:'test-identity-proof'})};
    if(url==='/auth/finish')return {ok:true,json:async()=>({ticket:'t'.repeat(32)})};
    if(String(url).endsWith('/sign-in/social'))return {ok:true,json:async()=>({url:SCENARIO==='evil'?'https://attacker.example/':CONFIG.provider+'/sign-in/social/init'})};
    return {ok:true,json:async()=>({})};
  }
};
vm.createContext(context);vm.runInContext(SCRIPT,context);
setImmediate(()=>console.log(JSON.stringify({calls,destinations,clean,notice:element('notice').textContent,progress:element('progress').hidden})));
"""
    source = (
        "const QUERY="
        + json.dumps(query)
        + ",SCENARIO="
        + json.dumps(scenario)
        + ",CONFIG="
        + json.dumps(config)
        + ",SCRIPT="
        + json.dumps(script.replace("__CONFIG__", json.dumps(config)))
        + ";\n"
        + harness
    )
    result = subprocess.run([node, "-e", source], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_oauth_verifier_is_exchanged_before_identity_proof_and_cleared():
    result = run_login("?flow=example&neon_auth_session_verifier=one-time-proof", "callback")
    paths = [c["url"] for c in result["calls"]]
    assert paths[0].endswith("/get-session?neon_auth_session_verifier=one-time-proof")
    assert paths.index("/auth/finish") > next(
        i for i, p in enumerate(paths) if p.endswith("/token")
    )
    assert "neon_auth_session_verifier" not in result["clean"][0]
    assert result["destinations"][0].startswith("https://ragbench.streamlit.app/?rb_flow=")


def test_failed_exchange_never_issues_app_session_or_redirect():
    result = run_login("?neon_auth_session_verifier=invalid", "failed")
    assert len(result["calls"]) == 1
    assert not result["destinations"]
    assert result["notice"]


@pytest.mark.parametrize("scenario,allowed", [("github", True), ("evil", False)])
def test_direct_github_entry_only_accepts_pinned_destinations(scenario, allowed):
    result = run_login("?method=github", scenario)
    assert len(result["destinations"]) == int(allowed)
    if allowed:
        assert (
            result["destinations"][0]
            == "https://test.neonauth.neon.tech/neondb/auth/sign-in/social/init"
        )
    else:
        assert result["notice"] == "Invalid sign-in destination"


@pytest.mark.parametrize(
    "scenario", ["prepare", "same_browser", "forwarded", "expired", "foreign_origin"]
)
def test_native_return_binding_requires_same_browser_and_does_not_rerun_in_a_loop(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser contract check")
    page = Path("dashboard/account_binding/index.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    source = (
        "const SCRIPT="
        + json.dumps(script)
        + ",SCENARIO="
        + json.dumps(scenario)
        + ";"
        + r"""
const vm=require('node:vm'), messages=[], store={};let listener;
Object.defineProperties(store,{
 getItem:{value:key=>store[key]||null},setItem:{value:(key,value)=>store[key]=value},
 removeItem:{value:key=>delete store[key]}
});
const flow='f'.repeat(32),verifier='v'.repeat(32),parent={postMessage:value=>messages.push(value)};
if(SCENARIO==='same_browser'||SCENARIO==='expired')store['rb_auth_flow_'+flow]=JSON.stringify({verifier,expires:SCENARIO==='expired'?0:700000});
const context={location:{origin:'https://ragbench.streamlit.app'},localStorage:store,
 Date:{now:()=>100000},window:{parent,addEventListener:(_,fn)=>listener=fn}};
vm.createContext(context);vm.runInContext(SCRIPT,context);
const event={source:parent,origin:SCENARIO==='foreign_origin'?'https://attacker.example':'https://ragbench.streamlit.app',
 data:{type:'streamlit:render',args:{mode:SCENARIO==='prepare'?'prepare':'redeem',flow,verifier}}};
listener(event);listener(event);
console.log(JSON.stringify(messages.filter(m=>m.type==='streamlit:setComponentValue').map(m=>m.value)));
"""
    )
    result = subprocess.run([node, "-e", source], capture_output=True, text=True, check=True)
    values = json.loads(result.stdout)
    if scenario == "foreign_origin":
        assert values == []
    elif scenario == "same_browser":
        assert values == [{"ready": True, "verifier": "v" * 32}]
    elif scenario == "prepare":
        assert values == [{"ready": True}]
    else:
        assert values == [{"ready": False}]
