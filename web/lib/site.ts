// Who built HeavDen Nexus and how to reach him: one place, so a future email or domain is a
// single edit.
export const BUILDER = {
  // the degree after a comma, without full stops, and never with "Dr" as well
  name: "Kelvin Ofori-Minta, PhD",
  title: "Data, ML & AI Engineer",
  brand: "HeavDen",
  email: "kelvinmeyet@gmail.com",
  linkedin: "https://www.linkedin.com/in/kelvin-ofori-minta-ph-d-9a5692b8",
};

export const MAILTO = `mailto:${BUILDER.email}?subject=${encodeURIComponent("About HeavDen Nexus")}`;

// The code is shared on request with recruiters and hiring teams, so the site never links to GitHub:
// every "code" link goes to the note on About me (CODE_PAGE), whose button opens this email.
export const CODE_PAGE = "/work-with-me#code";
export const CODE_REQUEST = `mailto:${BUILDER.email}?subject=${encodeURIComponent(
  "Code access request: HeavDen Nexus",
)}&body=${encodeURIComponent("Hi Kelvin,\n\nI'd like to look at the HeavDen Nexus code.\n\nName and company: ")}`;
