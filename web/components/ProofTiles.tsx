import styles from "./ProofTiles.module.css";

// The project's headline results, measured in production: version 3's test block (model card),
// prod's silver.vitals_reconciliation (20 Sep to 19 Oct, read 22 Oct) and the live site's red-team
// run (22 Oct). Keep them true: update them when those change.
const PROOF = [
  { figure: "22 of 28", label: "escalations flagged in advance in production, against 17 for the clinical standard" },
  { figure: "2,263,754", label: "monitor readings reconciled through the data platform, none lost" },
  { figure: "19 of 19", label: "adversarial tests passed by the AI assistant" },
  { figure: "< $25", label: "a month to keep the whole platform running, by design" },
];

/** Four tiles of glass for a dark band: two by two in a narrow panel, four across when wide. */
export function ProofTiles() {
  return (
    <section className={styles.panel} aria-labelledby="proof-title">
      <h2 id="proof-title" className="visually-hidden">
        Results from this project
      </h2>
      <dl className={styles.tiles}>
        {PROOF.map((p) => (
          <div key={p.figure}>
            <dt>{p.label}</dt>
            <dd>{p.figure}</dd>
          </div>
        ))}
      </dl>
      <p className={styles.note}>Measured on this project&rsquo;s synthetic data.</p>
    </section>
  );
}
