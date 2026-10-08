"use client";

import Link from "next/link";
import { useState } from "react";

import { BandTag, ErrorBox, Loading, RiskWithTrend } from "@/components/bits";
import { useApi } from "@/lib/api";
import { conditions, num, SITE_IDS, SITES, unitName, when } from "@/lib/format";
import type { Band, Board, SiteId } from "@/lib/types";

export default function PatientsPage() {
  const [site, setSite] = useState<SiteId | "">("");
  const [band, setBand] = useState<Band | "">("");
  const [limit, setLimit] = useState(40);
  const query = new URLSearchParams({ limit: String(limit) });
  if (site) query.set("site_id", site);
  if (band) query.set("band", band);
  const board = useApi<Board>(`/risk/board?${query}`);

  return (
    <div className="page">
      <header className="pageHead">
        <h1>Patients</h1>
        <p className="lede">
          Everyone on the wards{board.data ? ` at ${when(board.data.as_of)}` : ""}, highest risk first. Risk is
          the chance of needing the rapid response team or intensive care within six hours.
        </p>
      </header>

      <div className="filters">
        <label>
          Hospital{" "}
          <select className="select" value={site} onChange={(e) => setSite(e.target.value as SiteId | "")}>
            <option value="">All</option>
            {SITE_IDS.map((id) => (
              <option key={id} value={id}>
                {SITES[id].short}
              </option>
            ))}
          </select>
        </label>
        <label>
          Band{" "}
          <select className="select" value={band} onChange={(e) => setBand(e.target.value as Band | "")}>
            <option value="">All</option>
            <option value="High">High (alerting)</option>
            <option value="Low">Low</option>
          </select>
        </label>
      </div>

      <section className="section" aria-label="Ward list">
        {board.error && <ErrorBox what="the ward list" message={board.error} />}
        {!board.data && !board.error && <Loading lines={8} />}
        {board.data && board.data.patients.length === 0 && (
          <p className="notice">No patients match these filters. Choose another hospital or band.</p>
        )}
        {board.data && board.data.patients.length > 0 && (
          <div className="tableWrap">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Patient</th>
                  <th scope="col">Where</th>
                  <th scope="col">Band</th>
                  <th scope="col">Risk (6-hour change)</th>
                  <th scope="col" className="num">NEWS2</th>
                  <th scope="col">Main reason</th>
                </tr>
              </thead>
              <tbody>
                {board.data.patients.map((p) => (
                  <tr key={p.encounter_id}>
                    <th scope="row" style={{ fontWeight: 400 }}>
                      <Link href={`/patients/${p.encounter_id}`}>
                        <strong>{p.patient_label}</strong>
                      </Link>
                      <div className="muted small">
                        {p.age}
                        {p.sex}, {conditions(p.conditions)}
                      </div>
                    </th>
                    <td>{unitName(p.unit_id)}</td>
                    <td>
                      <BandTag band={p.risk_band} />
                    </td>
                    <td>
                      <RiskWithTrend risk={p.risk} before={p.risk_6h_ago} />
                    </td>
                    <td className="num">{num(p.news2_total)}</td>
                    <td className="small">
                      {p.top_factors[0]
                        ? `${p.top_factors[0].factor} (${p.top_factors[0].direction})`
                        : "–"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {board.data && board.data.patients.length === limit && limit < 400 && (
          <p>
            <button className="buttonQuiet" onClick={() => setLimit(limit + 40)}>
              Show 40 more patients
            </button>
          </p>
        )}
      </section>
    </div>
  );
}
