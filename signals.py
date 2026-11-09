"""
signals.py - THE ENGINE (Section 10.2). A background program, separate from
the dashboard, that does all the work and writes everything to the database.

    python signals.py            (stop with Ctrl+C)

Phase 1 (now) - it:
  * logs in to the IG demo account and streams all five instruments
    (prices + market status), reconnecting automatically (broker.py)
  * records every price and 1-minute bars to Parquet, and logs gaps
    (fetch_data.py), rewriting the day's files every RECORDER_FLUSH_SECONDS
  * writes the feed status and a heartbeat for the dashboard (db.py)
  * picks up kill-switch / halt requests from the jobs table
  * applies the automatic-halt rules (5 failed reconnects in a row,
    10 disconnections in 60 min, 5 failed IG requests in a row) and clears a
    daily-loss halt at the 17:00 New York reset
Later (Phase 2/3): build 5-minute bars, run every model variant that is not
Off, apply the filters and hard limits (Section 8), trade cards, demo orders.

Only one engine may run at a time: IG allows one streaming connection per
account. The engine refuses to start if another one's heartbeat is fresh.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler

import broker
import config
import db
import fetch_data
import timeutils

log = logging.getLogger("engine")


class Engine:
    def __init__(self):
        self.broker: broker.Broker | None = None
        self.stream: broker.PriceStream | None = None
        self.recorder: fetch_data.Recorder | None = None
        self.scheduler: BackgroundScheduler | None = None
        self._stopped = threading.Event()

    # ---------------- start / stop ----------------
    def start(self) -> None:
        db.init_db()
        if db.engine_is_running():
            raise RuntimeError(
                "Another engine seems to be running (its heartbeat is recent). Only one may run, "
                "because IG allows one streaming connection per account. If it crashed, wait "
                f"{config.ENGINE_HEARTBEAT_STALE_SECONDS} s and try again.")
        last_heartbeat = db.get_engine_status().get("heartbeat")

        self.broker = broker.Broker()
        self.broker.connect()
        log.info("Logged in to IG DEMO")
        self._store_market_info()

        self.recorder = fetch_data.Recorder()
        self.recorder.start(last_heartbeat=last_heartbeat)

        epics = [inst["epic"] for inst in config.INSTRUMENTS.values()]
        self.stream = broker.PriceStream(
            self.broker, epics,
            on_tick=self.recorder.on_tick,
            on_event=self._on_stream_event,
            on_market_status=self.recorder.on_market_status,
        )
        db.set_engine_status({"state": "running", "started_utc": timeutils.now_utc().isoformat(),
                              "stream_status": "starting"})
        self.stream.start()

        self.scheduler = BackgroundScheduler(timezone="UTC")
        common = dict(trigger="interval", max_instances=1, coalesce=True)
        self.scheduler.add_job(self._safe(self._maintenance), id="maintenance",
                               seconds=config.RECORDER_FLUSH_SECONDS, **common)
        self.scheduler.add_job(self._safe(self._check_halt), id="halt_check",
                               seconds=config.ENGINE_HALT_CHECK_SECONDS, **common)
        self.scheduler.add_job(self._safe(self._process_jobs), id="jobs",
                               seconds=config.ENGINE_JOB_POLL_SECONDS, **common)
        self.scheduler.add_job(self._safe(self._feed_status), id="feed_status",
                               seconds=config.FEED_STATUS_SECONDS, **common)
        self.scheduler.start()
        log.info("Engine running: recording %s. Files rewritten every %d s. Ctrl+C to stop.",
                 ", ".join(config.INSTRUMENTS), config.RECORDER_FLUSH_SECONDS)

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        log.info("Stopping the engine ...")
        for step, action in (
            ("scheduler", lambda: self.scheduler and self.scheduler.shutdown(wait=True)),
            ("stream", lambda: self.stream and self.stream.stop()),
            ("recorder", lambda: self.recorder and self.recorder.close()),
            ("logout", lambda: self.broker and self.broker.disconnect()),
            ("status", lambda: db.set_engine_status({"state": "stopped", "stream_status": "stopped"})),
        ):
            try:
                action()
            except Exception:
                log.exception("Error while stopping (%s) - continuing", step)
        log.info("Engine stopped.")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stopped.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Ctrl+C received")
        finally:
            self.stop()

    def _store_market_info(self) -> None:
        """Contract details (pip size, decimals, minimum deal...) for the dashboard and, later, sizing."""
        for key, inst in config.INSTRUMENTS.items():
            try:
                db.upsert_market_info(key, self.broker.get_market(inst["epic"]))
            except broker.BrokerError as exc:
                log.warning("Could not read market details for %s: %s", key, exc)

    # ---------------- scheduled work ----------------
    @staticmethod
    def _safe(func):
        """Never let one failing run kill the scheduler job; log it instead."""
        def wrapper():
            try:
                func()
            except Exception:
                log.exception("%s failed (will run again on schedule)", func.__name__)
        wrapper.__name__ = func.__name__
        return wrapper

    def _on_stream_event(self, kind: str, detail: str) -> None:
        if kind == "status" and self.recorder is not None and self.stream is not None:
            self.recorder.set_streaming(self.stream.is_streaming)

    def _maintenance(self) -> None:
        """Every RECORDER_FLUSH_SECONDS: write files and gaps, feed status, heartbeat."""
        summary = self.recorder.maintenance()
        health = self.stream.health()
        blocked = db.trading_block_reasons()
        db.set_engine_status({
            "state": "running",
            "stream_status": health["status"],
            "streaming": health["streaming"],
            "ticks_received": health["tick_count"],
            "connections_opened": health["connections_opened"],
            "market_status_error": self.stream.market_status_error or "",
            "trading_blocked": "; ".join(blocked),
        })
        feed = {f["instrument"]: f for f in db.get_feed_status()}
        parts = []
        for inst in config.INSTRUMENTS:
            f = feed.get(inst, {})
            last = f.get("last_tick_utc")
            when = timeutils.fmt_dual(last, seconds=True) if last else "no price yet"
            parts.append(f"{inst} {f.get('ticks_today', 0)} ({f.get('market_status') or '?'}, last {when})")
        log.info("Stream %s | %s | gaps open %d | dupes dropped %d%s",
                 health["status"], " | ".join(parts), summary["open_gaps"],
                 summary["duplicates_dropped"],
                 f" | TRADING BLOCKED: {'; '.join(blocked)}" if blocked else "")

    def _feed_status(self) -> None:
        """Every FEED_STATUS_SECONDS: latest prices for the dashboard's live table."""
        self.recorder.write_feed_status()

    def _check_halt(self) -> None:
        """Automatic-halt rules; the HALT stays on until cleared manually."""
        db.expire_halt()                     # daily-loss halt ends at 17:00 New York
        reason = self.stream.halt_reason() or self.broker.halt_reason()
        if reason and not db.get_flags()[db.HALT]["is_on"]:
            db.set_halt(reason, "engine")
            log.warning("HALT set automatically: %s", reason)

    def _process_jobs(self) -> None:
        """Requests from the dashboard (Phase 1 handles the kill switch and halt)."""
        while (job := db.claim_next_job()) is not None:
            kind, params = job["kind"], job["params"]
            reason = params.get("reason") or f"requested from the dashboard (job {job['id']})"
            try:
                if kind == db.JOB_KILL_SWITCH_ON:
                    db.set_kill_switch(reason, "dashboard")
                    log.warning("Kill switch ON (dashboard): %s", reason)
                elif kind == db.JOB_KILL_SWITCH_OFF:
                    db.clear_kill_switch(reason, "dashboard")
                    log.warning("Kill switch OFF (dashboard): %s", reason)
                elif kind == db.JOB_CLEAR_HALT:
                    db.clear_halt(reason, "dashboard")
                    log.warning("HALT cleared (dashboard): %s", reason)
                else:
                    db.finish_job(job["id"], ok=False, message=f"{kind} is not available yet (later phase)")
                    continue
                db.finish_job(job["id"], ok=True, message="done")
            except Exception as exc:
                db.finish_job(job["id"], ok=False, message=str(exc))
                log.exception("Job %s (%s) failed", job["id"], kind)


def main() -> int:
    timeutils.configure_logging(logging.INFO, log_file=config.LOGS_DIR / "engine.log")
    engine = Engine()
    try:
        engine.run_forever()
        return 0
    except broker.BrokerError as exc:
        log.error("Could not start: %s", exc)
        engine.stop()
        return 1
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
