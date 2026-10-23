import styles from "./PageHero.module.css";

/** Every page opens with this: a full-width dark band with the page's title and introduction on the
 *  left (about 70% of the screen) and, when there is one, a panel on the right (about 20%). */
export function PageHero({
  title,
  lockup,
  children,
  actions,
  aside,
}: {
  title: React.ReactNode;
  /** a line above the title, e.g. a name and a role */
  lockup?: React.ReactNode;
  /** the introduction: a `lede` paragraph and any notes */
  children?: React.ReactNode;
  /** buttons or links under the introduction */
  actions?: React.ReactNode;
  /** the right-hand panel */
  aside?: React.ReactNode;
}) {
  return (
    <header className={`nightBand ${styles.hero} ${aside ? styles.withAside : ""}`}>
      <div className={styles.text}>
        {lockup}
        <h1>{title}</h1>
        {children}
        {actions && <div className={styles.actions}>{actions}</div>}
      </div>
      {aside && <div className={styles.aside}>{aside}</div>}
    </header>
  );
}
