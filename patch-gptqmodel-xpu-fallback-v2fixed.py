from pathlib import Path
import gptqmodel

p = Path(gptqmodel.__file__).resolve().parent / "quantization" / "gptq.py"
text = p.read_text()

# Add a persistent per-module CPU-Hessian flag.
old = '''        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)

        self.validate_module(self.module)
'''

new = '''        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)

        # If Hessian accumulation OOMs on XPU for this module once,
        # route all remaining calibration batches directly to CPU.
        self._force_cpu_hessian = False

        self.validate_module(self.module)
'''

if old not in text:
    raise SystemExit("ERROR: init insertion point not found")

text = text.replace(old, new, 1)

# Replace the complete known-good v1 Hessian processing block.
old = '''        try:
            xtx = self.compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
        except RuntimeError as exc:
            if (
                torch.device(inp_device).type in ("cuda", "xpu")
                and "out of memory" in str(exc).lower()
            ):
                log.warn(
                    "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
                canonical_device = torch.device("cpu")
                self._final_hessian_device_hint = torch.device("cpu")
                xtx = self.compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp
'''

new = '''        if getattr(self, "_force_cpu_hessian", False):
            reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
            del reshaped_inp

            canonical_device = torch.device("cpu")
            self._final_hessian_device_hint = torch.device("cpu")

            xtx = self.compute_hessian_xtx(
                reshaped_inp_cpu
            ).to(dtype=torch.float32)

            xtx = xtx.detach()
            del reshaped_inp_cpu

        else:
            try:
                xtx = self.compute_hessian_xtx(
                    reshaped_inp
                ).to(dtype=torch.float32)

            except RuntimeError as exc:
                if (
                    torch.device(inp_device).type in ("cuda", "xpu")
                    and "out of memory" in str(exc).lower()
                ):
                    log.warn(
                        "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                        getattr(self, "name", "<unknown>"),
                    )

                    # Remember this decision for every remaining batch
                    # belonging to this GPTQ module.
                    self._force_cpu_hessian = True

                    reshaped_inp_cpu = reshaped_inp.to(
                        device=torch.device("cpu")
                    )
                    del reshaped_inp

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if (
                        hasattr(torch, "xpu")
                        and torch.xpu.is_available()
                    ):
                        torch.xpu.empty_cache()

                    canonical_device = torch.device("cpu")
                    self._final_hessian_device_hint = torch.device("cpu")

                    xtx = self.compute_hessian_xtx(
                        reshaped_inp_cpu
                    ).to(dtype=torch.float32)

                    xtx = xtx.detach()
                    del reshaped_inp_cpu

                else:
                    del reshaped_inp
                    raise

            else:
                xtx = xtx.detach()
                del reshaped_inp
'''

if old not in text:
    raise SystemExit("ERROR: v1 Hessian processing block not found")

text = text.replace(old, new, 1)
p.write_text(text)

# Compile the file now so the Docker build fails immediately on any syntax error.
compile(p.read_text(), str(p), "exec")

print("Patched:", p)
print("Python syntax: OK")
print("_force_cpu_hessian:", "_force_cpu_hessian" in p.read_text())
