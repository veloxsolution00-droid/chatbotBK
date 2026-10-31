"""Measure LAN upload staging, ingestion, chat latency, and local disk growth.
Usage: python benchmark_chatbot.py file1.pdf file2.docx --question 'What is this about?'
"""
import argparse, os, time, statistics, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

def folder_size(path):
    root=Path(path)
    if not root.exists(): return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--api", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    p.add_argument("--question", default="Summarize the main points in my uploaded documents.")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--concurrent", type=int, default=1, help="number of simultaneous chat requests for a simple load test")
    args=p.parse_args(); base=args.api.rstrip("/")
    for f in args.files:
        if not f.is_file(): p.error(f"File not found: {f}")
    total=sum(f.stat().st_size for f in args.files)
    print(f"Files: {len(args.files)} | Batch size: {total/1024/1024:.2f} MiB")
    health=requests.get(base+"/health", timeout=10); health.raise_for_status()
    h=health.json(); print("Limits:", h.get("limits"))
    headers = {}
    before=folder_size(h.get("data_path", "data"))
    files=[]
    handles=[]
    try:
        for f in args.files:
            handle=f.open("rb"); handles.append(handle)
            files.append(("files",(f.name,handle,"application/octet-stream")))
        t0=time.perf_counter(); r=requests.post(base+"/documents/upload",headers=headers,files=files,timeout=600); upload_seconds=time.perf_counter()-t0
        r.raise_for_status(); result=r.json()
    finally:
        for handle in handles: handle.close()
    print(f"Upload/staging HTTP time: {upload_seconds:.2f} s (server staging: {result.get('staging_seconds')} s)")
    print("Queued jobs:", ", ".join(x["job_id"] for x in result.get("results",[])))
    started=time.perf_counter(); jobs=[x["job_id"] for x in result.get("results",[])]; last={}
    while jobs and time.perf_counter()-started < args.timeout:
        done=[]
        for jid in jobs:
            q=requests.get(base+"/jobs/"+jid,headers=headers,timeout=10); q.raise_for_status(); job=q.json(); last[jid]=job
            if job["status"] in ("complete","failed"): done.append(jid)
        jobs=[j for j in jobs if j not in done]
        if jobs: time.sleep(2)
    ingestion_seconds=time.perf_counter()-started
    for jid,job in last.items(): print(f"Job {jid}: {job['status']} — {job.get('filename')}" + (f" — {job.get('error')}" if job.get('error') else ""))
    print(f"Time until all jobs completed/failed: {ingestion_seconds:.2f} s")
    if jobs: print("WARNING: timed out waiting for jobs:", jobs)
    if any(j.get("status")!="complete" for j in last.values()): print("WARNING: at least one ingestion job did not complete successfully.")
    def run_chat(i):
        t=time.perf_counter()
        response=requests.post(base+"/chat",headers=headers,json={"question":args.question},timeout=600)
        elapsed=time.perf_counter()-t; response.raise_for_status(); data=response.json()
        return i, elapsed, data
    count=max(1,min(args.concurrent,50))
    print(f"Running {count} simultaneous chat request(s)...")
    latencies=[]; server_latencies=[]; failures=[]; load_started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures=[pool.submit(run_chat,i+1) for i in range(count)]
        for future in as_completed(futures):
            try:
                i,elapsed,data=future.result(); latencies.append(elapsed)
                if isinstance(data.get("response_seconds"),(int,float)): server_latencies.append(data["response_seconds"])
                print(f"Question {i}: {elapsed:.2f} s (server: {data.get('response_seconds')} s; status: {data.get('status')})")
            except Exception as exc: failures.append(str(exc)); print("Chat request failed:",exc)
    print(f"Concurrent batch wall time: {time.perf_counter()-load_started:.2f} s; failures: {len(failures)}")
    after=folder_size(h.get("data_path", "data"))
    if latencies: print(f"Chat latency min/avg/max: {min(latencies):.2f}/{statistics.mean(latencies):.2f}/{max(latencies):.2f} s")
    if server_latencies: print(f"Server-measured latency min/avg/max: {min(server_latencies):.2f}/{statistics.mean(server_latencies):.2f}/{max(server_latencies):.2f} s")
    print(f"Application data/ folder size change: {(after-before)/1024/1024:.2f} MiB")
    print("For accurate total application storage, measure data/ folder size before and after the test.")

if __name__=="__main__": main()
