from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_SITES = [
    ("HOSP-A", "Regional Children's Hospital"),
    ("HOSP-B", "University Pediatric Medical Centre"),
    ("HOSP-C", "District Women and Children Hospital"),
    ("HOSP-D", "Community Pediatric Referral Hospital"),
]

TREATMENTS = [
    "Oral iron",
    "Iron plus nutrition counselling",
    "Iron plus deworming",
    "Intravenous iron",
    "Etiology-specific management",
]

SEX_VALUES = ["Female", "Male"]
RESPONSE_VALUES = ["Responder", "Partial responder", "Non-responder"]


@dataclass(frozen=True)
class Site:
    site_id: str
    site_name: str


@dataclass
class VisitRecord:
    record_id: str
    site_id: str
    patient_uid: str
    visit_number: int
    event_timestamp_utc: str
    age_months: int
    age_group: str
    sex: str
    weight_kg: float
    height_cm: float
    hb_g_dl: float
    rbc_million_ul: float
    pcv_pct: float
    mcv_fl: float
    mch_pg: float
    mchc_g_dl: float
    rdw_pct: float
    ferritin_ng_ml: float
    crp_mg_l: float
    reticulocyte_pct: float
    pallor_score: int
    symptom_score: int
    treatment: str
    dose_mg_day: float
    adherence_pct: Optional[float]
    nutrition_intervention: bool
    deworming: bool
    response_class: str
    hospitalization: bool
    referral: bool
    outcome: str
    data_source: str
    consent_recorded: bool
    validation_status: str
    sync_status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_patient_uid(site_id: str, local_patient_number: int, salt: str) -> str:
    """Create a deterministic, site-scoped pseudonymous identifier."""
    raw = f"{site_id}|{local_patient_number}|{salt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20].upper()


def classify_age(age_months: int) -> str:
    if 6 <= age_months <= 35:
        return "Infant/Toddler (6-35 months)"
    if age_months <= 71:
        return "Preschool (3-5 years)"
    if age_months <= 155:
        return "School-age (6-12 years)"
    return "Adolescent (13-17 years)"


def biological_limits() -> Dict[str, Tuple[float, float]]:
    """Broad plausibility ranges for data-quality screening, not diagnosis."""
    return {
        "age_months": (6, 215),
        "weight_kg": (3, 100),
        "height_cm": (50, 200),
        "hb_g_dl": (3, 20),
        "rbc_million_ul": (1, 8),
        "pcv_pct": (10, 60),
        "mcv_fl": (45, 110),
        "mch_pg": (12, 38),
        "mchc_g_dl": (20, 40),
        "rdw_pct": (10, 30),
        "ferritin_ng_ml": (1, 500),
        "crp_mg_l": (0, 200),
        "reticulocyte_pct": (0, 15),
        "pallor_score": (0, 4),
        "symptom_score": (0, 10),
        "dose_mg_day": (0, 300),
        "adherence_pct": (0, 100),
    }


def validate_record(record: VisitRecord) -> List[str]:
    errors: List[str] = []
    limits = biological_limits()
    values = asdict(record)

    required = [
        "record_id", "site_id", "patient_uid", "visit_number",
        "event_timestamp_utc", "age_months", "sex", "hb_g_dl",
        "treatment", "consent_recorded",
    ]
    for field in required:
        if values.get(field) in (None, ""):
            errors.append(f"missing:{field}")

    for field, (low, high) in limits.items():
        value = values.get(field)
        if value is None:
            continue
        if not (low <= float(value) <= high):
            errors.append(f"out_of_range:{field}")

    if record.visit_number < 1:
        errors.append("invalid:visit_number")
    if record.sex not in SEX_VALUES:
        errors.append("invalid:sex")
    if not record.consent_recorded:
        errors.append("invalid:consent_missing")

    return errors


def initialize_database(path: Path, central: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                record_id TEXT PRIMARY KEY,
                site_id TEXT NOT NULL,
                patient_uid TEXT NOT NULL,
                visit_number INTEGER NOT NULL,
                event_timestamp_utc TEXT NOT NULL,
                age_months INTEGER NOT NULL,
                age_group TEXT NOT NULL,
                sex TEXT NOT NULL,
                weight_kg REAL,
                height_cm REAL,
                hb_g_dl REAL,
                rbc_million_ul REAL,
                pcv_pct REAL,
                mcv_fl REAL,
                mch_pg REAL,
                mchc_g_dl REAL,
                rdw_pct REAL,
                ferritin_ng_ml REAL,
                crp_mg_l REAL,
                reticulocyte_pct REAL,
                pallor_score INTEGER,
                symptom_score INTEGER,
                treatment TEXT,
                dose_mg_day REAL,
                adherence_pct REAL,
                nutrition_intervention INTEGER,
                deworming INTEGER,
                response_class TEXT,
                hospitalization INTEGER,
                referral INTEGER,
                outcome TEXT,
                data_source TEXT,
                consent_recorded INTEGER,
                validation_status TEXT,
                sync_status TEXT,
                UNIQUE(site_id, patient_uid, visit_number)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                record_id TEXT,
                site_id TEXT,
                action TEXT,
                action_timestamp_utc TEXT,
                details_json TEXT
            )
        """)
        if central:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_rounds (
                    round_id INTEGER PRIMARY KEY,
                    aggregation_timestamp_utc TEXT,
                    participating_sites INTEGER,
                    total_local_samples INTEGER,
                    global_weights_json TEXT
                )
            """)
        conn.commit()


def insert_visit(db_path: Path, record: VisitRecord, errors: Sequence[str]) -> bool:
    payload = asdict(record)
    payload["validation_status"] = "VALID" if not errors else "REJECTED"
    payload["sync_status"] = "PENDING" if not errors else "NOT_ELIGIBLE"
    record.validation_status = payload["validation_status"]
    record.sync_status = payload["sync_status"]

    fields = list(payload.keys())
    placeholders = ",".join(["?"] * len(fields))
    values = [
        int(v) if isinstance(v, bool) else v
        for v in (payload[f] for f in fields)
    ]

    with sqlite3.connect(db_path) as conn:
        try:
            conn.execute(
                f"INSERT INTO visits ({','.join(fields)}) VALUES ({placeholders})",
                values,
            )
            action = "CREATE_VALID_RECORD" if not errors else "REJECT_RECORD"
            conn.execute(
                """INSERT INTO audit_log
                   (audit_id, record_id, site_id, action, action_timestamp_utc, details_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    record.record_id,
                    record.site_id,
                    action,
                    utc_now(),
                    json.dumps({"validation_errors": list(errors)}),
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.execute(
                """INSERT INTO audit_log
                   (audit_id, record_id, site_id, action, action_timestamp_utc, details_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    record.record_id,
                    record.site_id,
                    "DUPLICATE_BLOCKED",
                    utc_now(),
                    json.dumps({"visit_number": record.visit_number}),
                ),
            )
            conn.commit()
            return False


def simulate_patient_trajectory(
    site: Site,
    local_patient_number: int,
    visits: int,
    rng: random.Random,
    salt: str,
    study_start: datetime,
) -> Iterable[VisitRecord]:
    age_months = rng.randint(6, 215)
    patient_uid = stable_patient_uid(site.site_id, local_patient_number, salt)
    sex = rng.choice(SEX_VALUES)

    weight = max(4.0, round(0.24 * age_months + rng.uniform(2, 12), 1))
    height = min(190.0, round(60 + 0.58 * age_months + rng.uniform(-6, 6), 1))
    hb = round(rng.uniform(6.2, 10.9), 1)
    ferritin = round(rng.uniform(4, 25), 1)
    baseline_date = study_start + timedelta(days=rng.randint(0, 60))

    treatment = rng.choice(TREATMENTS)
    adherence = None

    for visit_number in range(1, visits + 1):
        event_date = baseline_date + timedelta(
            days=(visit_number - 1) * rng.choice([28, 30, 35, 42])
        )

        if visit_number > 1:
            adherence = round(rng.uniform(45, 100), 1)
            expected_gain = 0.15 + 0.007 * adherence
            hb = min(14.5, round(hb + rng.uniform(0.0, expected_gain), 1))
            ferritin = min(250, round(ferritin + rng.uniform(1, 15), 1))

        mcv = round(rng.uniform(58, 88) + (visit_number - 1) * rng.uniform(0, 1.8), 1)
        mch = round(rng.uniform(17, 29) + (visit_number - 1) * rng.uniform(0, 0.7), 1)
        mchc = round(rng.uniform(27, 35), 1)
        rbc = round(rng.uniform(3.0, 5.8), 2)
        pcv = round(max(18, min(48, hb * rng.uniform(2.7, 3.2))), 1)
        rdw = round(rng.uniform(13, 23), 1)
        crp = round(rng.uniform(0, 18), 1)
        retic = round(rng.uniform(0.5, 4.0), 1)

        symptom_score = max(0, min(10, round(9.5 - hb + rng.uniform(-1, 1))))
        pallor_score = max(0, min(4, round(4.5 - hb / 3 + rng.uniform(-0.5, 0.5))))

        if visit_number == 1:
            response = "Baseline"
        elif hb >= 11.0 or hb - 6.5 >= 2.0:
            response = "Responder"
        elif hb >= 9.5:
            response = "Partial responder"
        else:
            response = "Non-responder"

        hospitalization = hb < 6.5 and rng.random() < 0.45
        referral = hb < 7.0 or (response == "Non-responder" and visit_number >= 3)
        outcome = (
            "Recovered" if hb >= 11.5
            else "Improved" if hb >= 10.0
            else "Stable" if hb >= 8.0
            else "Needs specialist review"
        )

        dose = round(rng.uniform(20, 120), 1)
        record = VisitRecord(
            record_id=str(uuid.uuid4()),
            site_id=site.site_id,
            patient_uid=patient_uid,
            visit_number=visit_number,
            event_timestamp_utc=event_date.replace(tzinfo=timezone.utc).isoformat(),
            age_months=age_months,
            age_group=classify_age(age_months),
            sex=sex,
            weight_kg=weight,
            height_cm=height,
            hb_g_dl=hb,
            rbc_million_ul=rbc,
            pcv_pct=pcv,
            mcv_fl=mcv,
            mch_pg=mch,
            mchc_g_dl=mchc,
            rdw_pct=rdw,
            ferritin_ng_ml=ferritin,
            crp_mg_l=crp,
            reticulocyte_pct=retic,
            pallor_score=pallor_score,
            symptom_score=symptom_score,
            treatment=treatment,
            dose_mg_day=dose,
            adherence_pct=adherence,
            nutrition_intervention=rng.random() < 0.70,
            deworming=rng.random() < 0.35,
            response_class=response,
            hospitalization=hospitalization,
            referral=referral,
            outcome=outcome,
            data_source=rng.choice(
                ["Electronic case-report form", "Laboratory interface", "Point-of-care entry"]
            ),
            consent_recorded=True,
            validation_status="PENDING",
            sync_status="PENDING",
        )
        yield record


def synchronize_site_to_central(site_db: Path, central_db: Path, site_id: str) -> int:
    """Copy valid, unsynchronized, de-identified visits to central storage."""
    with sqlite3.connect(site_db) as src, sqlite3.connect(central_db) as dst:
        src.row_factory = sqlite3.Row
        rows = src.execute(
            """SELECT * FROM visits
               WHERE validation_status='VALID' AND sync_status='PENDING'
               ORDER BY event_timestamp_utc"""
        ).fetchall()

        inserted = 0
        for row in rows:
            fields = row.keys()
            placeholders = ",".join(["?"] * len(fields))
            try:
                dst.execute(
                    f"INSERT INTO visits ({','.join(fields)}) VALUES ({placeholders})",
                    tuple(row[field] for field in fields),
                )
                src.execute(
                    "UPDATE visits SET sync_status='SYNCED' WHERE record_id=?",
                    (row["record_id"],),
                )
                src.execute(
                    """INSERT INTO audit_log
                       (audit_id, record_id, site_id, action, action_timestamp_utc, details_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        row["record_id"],
                        site_id,
                        "SYNC_TO_CENTRAL",
                        utc_now(),
                        json.dumps({"central_database": central_db.name}),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                src.execute(
                    "UPDATE visits SET sync_status='DUPLICATE_AT_CENTRAL' WHERE record_id=?",
                    (row["record_id"],),
                )

        src.commit()
        dst.commit()
        return inserted


def local_model_update(site_db: Path, rng: random.Random, dimensions: int = 8) -> Tuple[List[float], int]:
    """
    Simulate a local model update. This is an operational placeholder for
    hospital-side PAF training; no raw patient rows leave the site.
    """
    with sqlite3.connect(site_db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM visits WHERE validation_status='VALID'"
        ).fetchone()[0]
    weights = [round(rng.uniform(-0.25, 0.25), 6) for _ in range(dimensions)]
    return weights, n


def federated_average(updates: Sequence[Tuple[List[float], int]]) -> List[float]:
    total = sum(n for _, n in updates)
    if total == 0:
        raise ValueError("No local samples available for aggregation.")
    dimensions = len(updates[0][0])
    return [
        round(sum(weights[j] * n for weights, n in updates) / total, 6)
        for j in range(dimensions)
    ]


def store_federated_round(
    central_db: Path,
    round_id: int,
    updates: Sequence[Tuple[List[float], int]],
    global_weights: Sequence[float],
) -> None:
    with sqlite3.connect(central_db) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO federated_rounds
               (round_id, aggregation_timestamp_utc, participating_sites,
                total_local_samples, global_weights_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                round_id,
                utc_now(),
                len(updates),
                sum(n for _, n in updates),
                json.dumps(list(global_weights)),
            ),
        )
        conn.commit()


def export_outputs(output_dir: Path, central_db: Path, site_dbs: Dict[str, Path]) -> None:
    with sqlite3.connect(central_db) as conn:
        central_df = pd.read_sql_query(
            "SELECT * FROM visits ORDER BY site_id, patient_uid, visit_number", conn
        )
        rounds_df = pd.read_sql_query(
            "SELECT * FROM federated_rounds ORDER BY round_id", conn
        )

    site_summary_rows = []
    for site_id, db_path in site_dbs.items():
        with sqlite3.connect(db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
            valid = conn.execute(
                "SELECT COUNT(*) FROM visits WHERE validation_status='VALID'"
            ).fetchone()[0]
            synced = conn.execute(
                "SELECT COUNT(*) FROM visits WHERE sync_status='SYNCED'"
            ).fetchone()[0]
            rejected = conn.execute(
                "SELECT COUNT(*) FROM visits WHERE validation_status='REJECTED'"
            ).fetchone()[0]
        site_summary_rows.append(
            {
                "site_id": site_id,
                "total_local_records": total,
                "valid_records": valid,
                "synced_records": synced,
                "rejected_records": rejected,
                "sync_completeness_pct": round(100 * synced / valid, 2) if valid else 0,
            }
        )

    site_summary_df = pd.DataFrame(site_summary_rows)

    metadata_df = pd.DataFrame(
        [
            ["Dataset status", "Synthetic multisite operational simulation"],
            ["Population", "Pediatric participants aged 6 months to 17 years"],
            ["Design", "Prospective longitudinal, repeated visits"],
            ["Data management", "Local SQLite plus de-identified central synchronization"],
            ["Privacy", "Site-scoped SHA-256 pseudonymous patient identifiers"],
            ["Quality control", "Required fields, range checks, duplicate blocking, audit logs"],
            ["Federated learning", "Simulated parameter-only weighted aggregation"],
            ["Clinical use", "Not approved for clinical inference"],
        ],
        columns=["Characteristic", "Description"],
    )

    central_df.to_csv(output_dir / "central_longitudinal_dataset.csv", index=False)

    excel_path = output_dir / "multisite_realtime_pediatric_anemia_dataset.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        central_df.to_excel(writer, sheet_name="Longitudinal_Records", index=False)
        site_summary_df.to_excel(writer, sheet_name="Site_Quality_Summary", index=False)
        rounds_df.to_excel(writer, sheet_name="Federated_Rounds", index=False)
        metadata_df.to_excel(writer, sheet_name="Dataset_Metadata", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column
                )
                sheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate real-time multisite pediatric anemia data collection."
    )
    parser.add_argument("--patients-per-site", type=int, default=25)
    parser.add_argument("--visits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--salt", default="REPLACE_WITH_SECURE_SITE_SECRET")
    parser.add_argument("--output-dir", default="multisite_paf_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patients_per_site < 1 or args.visits < 1:
        raise ValueError("patients-per-site and visits must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sites = [Site(*site) for site in DEFAULT_SITES]
    central_db = output_dir / "central_research_database.sqlite"
    initialize_database(central_db, central=True)

    site_dbs: Dict[str, Path] = {}
    study_start = datetime(2025, 1, 1)

    for site in sites:
        site_db = output_dir / f"{site.site_id.lower()}_local.sqlite"
        initialize_database(site_db)
        site_dbs[site.site_id] = site_db

        for patient_number in range(1, args.patients_per_site + 1):
            for record in simulate_patient_trajectory(
                site=site,
                local_patient_number=patient_number,
                visits=args.visits,
                rng=rng,
                salt=args.salt,
                study_start=study_start,
            ):
                errors = validate_record(record)
                insert_visit(site_db, record, errors)

        synchronized = synchronize_site_to_central(
            site_db=site_db,
            central_db=central_db,
            site_id=site.site_id,
        )
        print(f"{site.site_id}: synchronized {synchronized} valid records.")

    updates = [local_model_update(path, rng) for path in site_dbs.values()]
    global_weights = federated_average(updates)
    store_federated_round(central_db, round_id=1, updates=updates, global_weights=global_weights)

    export_outputs(output_dir, central_db, site_dbs)

    manifest = {
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "sites": [asdict(site) for site in sites],
        "patients_per_site": args.patients_per_site,
        "visits_per_patient": args.visits,
        "expected_records": len(sites) * args.patients_per_site * args.visits,
        "warning": "Synthetic operational simulation; no real patient data.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Completed. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()