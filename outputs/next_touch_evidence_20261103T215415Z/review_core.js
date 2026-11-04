(function (root) {
  "use strict";
  const statuses = ["EVENT_PRESENT", "NO_EVENT_VERIFIED", "UNCERTAIN", "NO_PLAYER_VISIBLE", "BAD_SYNCHRONIZATION", "OTHER"];
  function validate(row, item) {
    if (!row.reviewer || !statuses.includes(row.status)) throw Error("Reviewer and valid status required");
    if (row.case_id !== item.case_id) throw Error("Case mismatch");
    if (row.status === "NO_EVENT_VERIFIED" && item.sync_verified !== true) throw Error("Independent synchronization is required for verified negatives");
    if (row.status === "EVENT_PRESENT" && !row.event_type) throw Error("Event type required");
    if (!["LOW", "MEDIUM", "HIGH"].includes(row.confidence)) throw Error("Confidence required");
  }
  function record(labels, item, input) {
    validate(input, item);
    if (labels.some(x => x.case_id === input.case_id && x.reviewer === input.reviewer)) throw Error("First judgment is immutable; use separate arbitration");
    const row = {...input, evidence_kind: item.evidence_kind, sync_verified: item.sync_verified === true, first_label_utc: new Date().toISOString(), prediction_exposed_before_first_label: false};
    return labels.concat([Object.freeze(row)]);
  }
  function reveal(labels, item, reviewer) {
    if (!labels.some(x => x.case_id === item.case_id && x.reviewer === reviewer)) throw Error("Save a first judgment before revealing");
    return item.hidden_predictions || {};
  }
  function merge(labels, imported, queue) {
    let result = labels.slice();
    for (const row of imported) {
      const item = queue.find(x => x.case_id === row.case_id);
      if (!item) throw Error("Unknown imported case");
      validate(row, item);
      if (row.prediction_exposed_before_first_label !== false || !row.first_label_utc) throw Error("Unverified blinding provenance");
      const duplicate = result.find(x => x.case_id === row.case_id && x.reviewer === row.reviewer);
      if (duplicate) {
        if (JSON.stringify(duplicate) !== JSON.stringify(row)) throw Error("Conflicting immutable first judgment");
      } else result.push(Object.freeze({...row}));
    }
    return result;
  }
  function csv(labels) {
    const keys = ["case_id","reviewer","status","actor","event_type","time_window","confidence","notes","evidence_kind","sync_verified","first_label_utc","prediction_exposed_before_first_label"];
    const quote = value => '"' + String(value == null ? "" : value).replace(/"/g, '""') + '"';
    return [keys.map(quote).join(",")].concat(labels.map(row => keys.map(k => quote(row[k])).join(","))).join("\r\n");
  }
  function arbitrate(arbitrations, labels, caseId, reviewer, conclusion, notes) {
    if (new Set(labels.filter(x => x.case_id === caseId).map(x => x.reviewer)).size < 2) throw Error("Two real independent reviewer records required");
    if (!reviewer || !conclusion) throw Error("Arbitrator and conclusion required");
    return arbitrations.concat([{case_id:caseId, arbitrator:reviewer, conclusion, notes, utc:new Date().toISOString(), originals_preserved:true}]);
  }
  root.ReviewCore = {statuses, record, reveal, merge, csv, arbitrate};
  if (typeof module !== "undefined") module.exports = root.ReviewCore;
})(typeof globalThis !== "undefined" ? globalThis : this);
