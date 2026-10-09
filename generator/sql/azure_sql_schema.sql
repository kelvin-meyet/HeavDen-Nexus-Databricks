-- HeavDen Health: hospital source-of-record schema (Azure SQL Database, T-SQL).
-- Synthetic data only. Loaded by generator/source_db.py from Synthea and the hospital simulation.
-- One database holds every environment: generator/source_db.py replaces `dbo.` with the
-- environment's schema (dev, staging, prod) before running this file.
-- Databricks ingests with a serverless JDBC job that reads SQL Server Change Tracking
-- (Plan.md §7.2). last_updated holds the *simulated* time of the row's latest change
-- (admission, transfer, discharge), which Silver uses to order SCD2 history.

ALTER DATABASE CURRENT
SET CHANGE_TRACKING = ON (CHANGE_RETENTION = 7 DAYS, AUTO_CLEANUP = ON);
GO

CREATE TABLE dbo.sites (
    site_id      VARCHAR(16)   NOT NULL PRIMARY KEY,
    name         NVARCHAR(100) NOT NULL,
    city         NVARCHAR(60)  NOT NULL,
    beds         INT           NOT NULL,
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
);

CREATE TABLE dbo.units (
    unit_id      VARCHAR(40)  NOT NULL PRIMARY KEY,
    site_id      VARCHAR(16)  NOT NULL REFERENCES dbo.sites (site_id),
    unit_type    VARCHAR(20)  NOT NULL,  -- general | step_down | respiratory
    beds         INT          NOT NULL,
    last_updated DATETIME2    NOT NULL DEFAULT SYSUTCDATETIME()
);

-- PII-like columns (names, birth_date) get Unity Catalog column masks downstream.
CREATE TABLE dbo.patients (
    patient_id   VARCHAR(36)   NOT NULL PRIMARY KEY,  -- Synthea UUID
    first_name   NVARCHAR(60)  NOT NULL,
    last_name    NVARCHAR(60)  NOT NULL,
    birth_date   DATE          NOT NULL,
    sex          CHAR(1)       NOT NULL,
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
);

CREATE TABLE dbo.conditions (
    patient_id   VARCHAR(36)   NOT NULL REFERENCES dbo.patients (patient_id),
    code         VARCHAR(20)   NOT NULL,  -- SNOMED-CT
    description  NVARCHAR(500) NOT NULL,
    start_date   DATE          NOT NULL,
    stop_date    DATE          NULL,      -- NULL = still active
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_conditions PRIMARY KEY (patient_id, code, start_date)
);

CREATE TABLE dbo.medications (
    patient_id   VARCHAR(36)   NOT NULL REFERENCES dbo.patients (patient_id),
    code         VARCHAR(20)   NOT NULL,  -- RxNorm
    description  NVARCHAR(500) NOT NULL,
    start_ts     DATETIME2     NOT NULL,
    stop_ts      DATETIME2     NULL,
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_medications PRIMARY KEY (patient_id, code, start_ts)
);

-- Current and past hospital stays. Admissions, transfers and discharges update this table
-- during the simulation; last_updated changes, so the next ingestion run picks them up.
CREATE TABLE dbo.encounters (
    encounter_id VARCHAR(16)   NOT NULL PRIMARY KEY,
    patient_id   VARCHAR(36)   NOT NULL REFERENCES dbo.patients (patient_id),
    site_id      VARCHAR(16)   NOT NULL REFERENCES dbo.sites (site_id),
    unit_id      VARCHAR(40)   NOT NULL REFERENCES dbo.units (unit_id),
    bed_id       VARCHAR(48)   NOT NULL,
    admit_ts     DATETIME2     NOT NULL,
    discharge_ts DATETIME2     NULL,      -- NULL = still in hospital
    status       VARCHAR(12)   NOT NULL,  -- admitted | discharged
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
);

-- Nurse-charted observations (insert-only): consciousness (ACVPU) and supplemental oxygen.
-- These complete NEWS2; devices can't measure them.
CREATE TABLE dbo.nurse_observations (
    encounter_id VARCHAR(16)   NOT NULL REFERENCES dbo.encounters (encounter_id),
    patient_id   VARCHAR(36)   NOT NULL REFERENCES dbo.patients (patient_id),
    obs_ts       DATETIME2     NOT NULL,
    acvpu        CHAR(1)       NOT NULL,  -- A | C | V | P | U
    on_oxygen    BIT           NOT NULL,
    o2_flow_lpm  DECIMAL(4,1)  NOT NULL,
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_nurse_observations PRIMARY KEY (encounter_id, obs_ts)
);

-- Which monitor is on which patient, and when. Vitals events carry device_id only;
-- Silver joins them to patient_id through this table.
CREATE TABLE dbo.device_assignments (
    device_id    VARCHAR(16)   NOT NULL,
    encounter_id VARCHAR(16)   NOT NULL REFERENCES dbo.encounters (encounter_id),
    patient_id   VARCHAR(36)   NOT NULL REFERENCES dbo.patients (patient_id),
    start_ts     DATETIME2     NOT NULL,
    end_ts       DATETIME2     NULL,
    last_updated DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT pk_device_assignments PRIMARY KEY (device_id, start_ts)
);
GO

ALTER TABLE dbo.sites              ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.units              ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.patients           ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.conditions         ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.medications        ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.encounters         ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.device_assignments ENABLE CHANGE_TRACKING;
ALTER TABLE dbo.nurse_observations ENABLE CHANGE_TRACKING;
GO

-- Cursor-column indexes so the connector's "WHERE last_updated > @cursor" queries stay cheap.
CREATE INDEX ix_sites_last_updated              ON dbo.sites (last_updated);
CREATE INDEX ix_units_last_updated              ON dbo.units (last_updated);
CREATE INDEX ix_patients_last_updated           ON dbo.patients (last_updated);
CREATE INDEX ix_conditions_last_updated         ON dbo.conditions (last_updated);
CREATE INDEX ix_medications_last_updated        ON dbo.medications (last_updated);
CREATE INDEX ix_encounters_last_updated         ON dbo.encounters (last_updated);
CREATE INDEX ix_device_assignments_last_updated ON dbo.device_assignments (last_updated);
CREATE INDEX ix_nurse_observations_last_updated ON dbo.nurse_observations (last_updated);
GO
