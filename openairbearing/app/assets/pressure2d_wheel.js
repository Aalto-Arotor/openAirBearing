(function () {
  const GRAPH_IDS = ["pressure-2d-plot-static", "pressure-2d-plot-moving"];
  const SLIDER_ID = "pressure-2d-slider";
  const WHEEL_THRESHOLD = 10; // larger = less sensitive (try 120, 180, 240)

  let wheelAcc = 0;
  let cameraSyncLock = false;

  function bindCameraSync() {
    for (const sourceId of GRAPH_IDS) {
      const sourceHost = document.getElementById(sourceId);
      if (!sourceHost) continue;

      const sourceGd = sourceHost.querySelector(".js-plotly-plot");
      if (!sourceGd || sourceGd.dataset.cameraSyncBound === "1") continue;

      sourceGd.dataset.cameraSyncBound = "1";
      sourceGd.on("plotly_relayout", function (eventData) {
        if (cameraSyncLock) return;

        const camera = eventData?.["scene.camera"];
        if (!camera) return;

        cameraSyncLock = true;
        try {
          for (const targetId of GRAPH_IDS) {
            if (targetId === sourceId) continue;

            const targetHost = document.getElementById(targetId);
            const targetGd = targetHost?.querySelector(".js-plotly-plot");
            if (!targetGd) continue;

            if (window.Plotly?.relayout) {
              window.Plotly.relayout(targetGd, { "scene.camera": camera });
            }
          }
        } finally {
          setTimeout(() => {
            cameraSyncLock = false;
          }, 0);
        }
      });
    }
  }

  function bindWheel() {
    for (const graphId of GRAPH_IDS) {
      const host = document.getElementById(graphId);
      if (!host) continue;

      const gd = host.querySelector(".js-plotly-plot");
      if (!gd || gd.dataset.wheelBound === "1") continue;

      gd.dataset.wheelBound = "1";
      gd.addEventListener(
        "wheel",
        function (e) {
          const sliderRoot = document.getElementById(SLIDER_ID);
          const handle = sliderRoot?.querySelector(".rc-slider-handle");
          if (!handle) return;

          const cur = Number(handle.getAttribute("aria-valuenow"));
          const min = Number(handle.getAttribute("aria-valuemin"));
          const max = Number(handle.getAttribute("aria-valuemax"));
          if (!Number.isFinite(cur) || !Number.isFinite(min) || !Number.isFinite(max)) return;

          e.preventDefault();
          e.stopPropagation();

          wheelAcc += e.deltaY;

          let step = 0;
          while (Math.abs(wheelAcc) >= WHEEL_THRESHOLD) {
            step += Math.sign(wheelAcc);
            wheelAcc -= Math.sign(wheelAcc) * WHEEL_THRESHOLD;
          }
          if (step === 0) return;

          const next = Math.max(min, Math.min(max, cur + step));
          if (next === cur) return;

          if (window.dash_clientside?.set_props) {
            window.dash_clientside.set_props(SLIDER_ID, { value: next });
          }
        },
        { passive: false, capture: true }
      );
    }
  }

  function bindAll() {
    bindWheel();
    bindCameraSync();
  }

  setInterval(bindAll, 400);
})();